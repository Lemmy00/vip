import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoImageProcessor
from typing import Union, List


class DistanceModel(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        dino_model_name: str = 'dinov2-base',
        freeze_dino: bool = False,
        decoder_layers: int = 3,
        nhead: int = 8,
        dist_type: str = "linear",
        output_dim: int = 1,
        lr: float = 1e-4,
        num_negatives: int = 3,
        gamma: float = 0.98,
        l2weight: float = 0.001,
        l1weight: float = 0.001
    ):
        """
        Initializes the DistanceModel with an encoder (DINOv2) and a Transformer decoder.
        
        Args:
            device (str): Device to run the model.
            dino_model_name (str): Name of the DINOv2 model variant.
            freeze_dino (bool): Whether to freeze the encoder parameters.
            decoder_layers (int): Number of layers in the Transformer decoder.
            nhead (int): Number of attention heads in the decoder.
            dist_type (str): Type of the final prediction head ("linear" or "l2").
            output_dim (int): Dimension of the output from the prediction head.
            lr (float): Learning rate for optimizers.
            num_negatives (int): (Unused in this snippet) number of negatives.
            gamma (float): (Unused in this snippet) gamma parameter.
            l2weight (float): (Unused in this snippet) l2 regularization weight.
        """
        super().__init__()
        
        # Load the image processor and the DINO model from Hugging Face.
        hf_model_id = f"facebook/{dino_model_name}"
        self.processor = AutoImageProcessor.from_pretrained(hf_model_id, use_fast=False)
        self.tensor_processor = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.dino = AutoModel.from_pretrained(hf_model_id)
        
        # Optionally freeze the encoder's (DINO's) parameters.
        if freeze_dino:
            for param in self.dino.parameters():
                param.requires_grad = False

        # Determine the embedding dimension using a dummy forward pass.
        self.embed_dim = self._get_embed_dim()
        self.output_dim = output_dim
        
        # Build the Transformer decoder.
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        
        # Build the prediction head.
        self.dist_type = dist_type
        if self.dist_type == "linear":
            self.head = nn.Sequential(
                nn.Linear(self.embed_dim, self.output_dim),
                nn.Softplus()
            )
        elif self.dist_type == "l2":
            # For L2, the head is an identity and the final distance is computed via norm.
            self.head = nn.Identity()
        else:
            raise ValueError(f"Unsupported head type: {dist_type}")

        # Split the optimizers: one for the encoder and one for the decoder & head.
        self.encoder_opt = torch.optim.AdamW(self.dino.parameters(), lr=lr)
        self.decoder_opt = torch.optim.AdamW(
            list(self.decoder.parameters()) + list(self.head.parameters()), lr=lr
        )

        self.num_negatives = num_negatives
        self.gamma = gamma
        self.l2weight = l2weight
        self.l1weight = l1weight

    def _get_embed_dim(self) -> int:
        """
        Pass a dummy image through the DINO encoder to determine the embedding dimension.
        The encoder output is assumed to be a dictionary with key 'last_hidden_state'
        of shape [B, num_tokens, embed_dim].
        """
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            tokens = self.dino(dummy)['last_hidden_state']
        return tokens.shape[-1]
    
    def preprocess(self, images: Union[list, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess the input images using either a list of images or tensors.
        
        Args:
            images (Union[list, PIL.Image.Image, torch.Tensor]): One or more images.
        
        Returns:
            torch.Tensor: A batch of processed images.
        """
        if torch.is_tensor(images):
            return self.tensor_processor(images.to(torch.float32) / 255.0)
        elif isinstance(images, list) or isinstance(images, Image.Image):
            processed = self.processor(images=images, return_tensors="pt")
            return processed["pixel_values"]
        raise ValueError("Unrecognized image type given for processing.")
    
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encodes an image using the DINO encoder.
        
        Args:
            image (torch.Tensor): Image tensor of shape [B, 3, H, W].
        
        Returns:
            torch.Tensor: Encoder output tokens of shape [B, num_tokens, embed_dim].
        """
        # Return the 'last_hidden_state' tokens.
        return self.dino(image)['last_hidden_state']

    def decode(self, tokens_I: torch.Tensor, tokens_G: torch.Tensor) -> torch.Tensor:
        """
        Decodes the tokens using a Transformer decoder.
        
        Args:
            tokens_I (torch.Tensor): Encoder tokens from the query image; shape [B, num_tokens, embed_dim].
            tokens_G (torch.Tensor): Encoder tokens from the key/value image; shape [B, num_tokens, embed_dim].
        
        Returns:
            torch.Tensor: The output prediction (typically using the CLS token after decoding).
        """
        # Transformer expects input shape [sequence_length, batch_size, embed_dim].
        tokens_I = tokens_I.transpose(0, 1)
        tokens_G = tokens_G.transpose(0, 1)
        
        decoder_out = self.decoder(tgt=tokens_I, memory=tokens_G)
        cls_token = decoder_out[0]  # shape: [B, embed_dim]
        output = self.head(cls_token)
        return output

    def forward(self, image_I: torch.Tensor, image_G: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that processes two images: first encode then decode.
        
        Args:
            image_I (torch.Tensor): Query image tensor; shape: [B, 3, H, W].
            image_G (torch.Tensor): Key/Value image tensor; shape: [B, 3, H, W].
        
        Returns:
            torch.Tensor: The final output prediction.
        """
        # Encode both images.
        tokens_I = self.encode(image_I)
        tokens_G = self.encode(image_G)
        
        # Now decode the tokens.
        output = self.decode(tokens_I, tokens_G)
        return output

    def sim_ete(self, image_I: torch.Tensor, image_G: torch.Tensor) -> torch.Tensor:
        """
        Computes the similarity between two images based on the output prediction.
        
        Args:
            image_I (torch.Tensor): Query image tensor.
            image_G (torch.Tensor): Key/Value image tensor.
        
        Returns:
            torch.Tensor: A tensor of similarity scores for each batch element.
        """
        goal_dist = self.forward(image_I, image_G)
        if self.dist_type == "linear":
            # For linear head, apply a squeeze and then negate the distance.
            return -goal_dist.squeeze(1)
        elif self.dist_type == "l2":
            # For the l2 case, compute the norm.
            return -torch.linalg.norm(goal_dist, dim=-1)

    def sim(self, goal_dist: torch.Tensor) -> torch.Tensor:
        """
        Computes the similarity given the already computed dense distance.
        
        Args:
            goal_dist (torch.Tensor): The precomputed dense distance.
        
        Returns:
            torch.Tensor: The similarity scores.
        """
        if self.dist_type == "linear":
            return -goal_dist
        elif self.dist_type == "l2":
            return -torch.linalg.norm(goal_dist, dim=-1)

    def dense_dist(self, image_I: torch.Tensor, image_G: torch.Tensor) -> torch.Tensor:
        """
        Returns the encoded dense representation (using the CLS token of the decoder output)
        from the images.
        
        Args:
            image_I (torch.Tensor): Query image tensor.
            image_G (torch.Tensor): Key/Value image tensor.
        
        Returns:
            torch.Tensor: The dense distance representation.
        """
        # Encode both images.
        tokens_I = self.encode(image_I)
        tokens_G = self.encode(image_G)
        
        # Transform and decode; extract the CLS token.
        tokens_I = tokens_I.transpose(0, 1)
        tokens_G = tokens_G.transpose(0, 1)
        decoder_out = self.decoder(tgt=tokens_I, memory=tokens_G)
        cls_token = decoder_out[0]  # shape: [B, embed_dim]
        return cls_token


# Example usage:
if __name__ == "__main__":
    # Instantiate the model. You can now easily freeze or train the encoder separately.
    model = DistanceModel(dist_type="l2", freeze_dino=False)
    batch_size = 2

    # Create dummy input images (assuming 3x224x224 tensors).
    image_I = torch.randn(batch_size, 3, 224, 224)
    image_G = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass through the combined encoder + decoder.
    output = model(image_I, image_G)
    print("Output shape:", output.shape)
    
    # Compute similarity between images.
    similarity_scores = model.sim_ete(image_I, image_G)
    print("Similarity scores:", similarity_scores)

    similarity_scores2 = model.sim(output)
    print("Similarity scores (from output):", similarity_scores2)
    
    # If you wish to get the encoder outputs first, you can do so:
    enc_I = model.encode(image_I)
    enc_G = model.encode(image_G)
    print("Encoder output shape (Image I):", enc_I.shape)
    
    # Then later run the decoder:
    decoded_output = model.decode(enc_I, enc_G)
    print("Decoded output shape:", decoded_output.shape)
