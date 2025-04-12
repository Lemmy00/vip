import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL

from transformers import AutoModel, AutoImageProcessor
from torchvision import transforms
from PIL import Image
from typing import Union, List


class DistanceModel(nn.Module):
    def __init__(
        self,
        device="cuda",
        freeze_dino: bool = True,
        dino_model_name: str = 'dinov2-base',
        decoder_layers: int = 3,
        nhead: int = 8,
        dist_type: str = "linear",
        lr: float = 1e-4,
        num_negatives: int = 3,
        gamma: float = 0.98
    ):
        """
        Args:
            freeze_dino (bool): Whether to freeze the DINOv2 encoder weights.
            dino_model_name (str): Name of the DINOv2 model variant (e.g., 'dinov2_vitb14').
            decoder_layers (int): Number of Transformer decoder layers.
            nhead (int): Number of attention heads for the decoder.
            dist_type (str): Type of the final prediction head.
            lr (float): learning rate
        """
        super().__init__()
        
        # Load the auto image processor and model from Hugging Face.
        # The pre-trained model identifier is constructed using the provided dino_model_name.
        hf_model_id = f"facebook/{dino_model_name}"
        self.processor = AutoImageProcessor.from_pretrained(hf_model_id, use_fast=False)
        self.tensor_processor = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.dino = AutoModel.from_pretrained(hf_model_id)
        
        if freeze_dino:
            for param in self.dino.parameters():
                param.requires_grad = False

        # Determine the embedding dimension using a dummy forward pass.
        self.embed_dim = self._get_embed_dim()
        
        # Build the Transformer decoder.
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # Distance prediction operating on the output from the model.
        self.dist_type = dist_type
        if self.dist_type == "linear":
            self.head = nn.Sequential(nn.Linear(self.embed_dim, 1), 
                                      nn.Softplus())
        elif self.dist_type == "l2":
            self.head = nn.Identity()
        else:
            raise ValueError(f"Unsupported head type: {dist_type}")

        params = list(self.parameters()) 
        self.encoder_opt = torch.optim.AdamW(params, lr=lr)
        self.num_negatives = num_negatives
        self.gamma = gamma

    def _get_embed_dim(self):
        """
        Runs a dummy image through the DINOv2 encoder to
        determine the embedding dimension. The encoder output is assumed to be
        a dictionary with key 'last_hidden_state' shaped [B, num_tokens, embed_dim].
        """
        # Create a dummy image tensor of shape [1, 3, 224, 224].
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            tokens = self.dino(dummy)['last_hidden_state']
        return tokens.shape[-1]

    def preprocess(self, images: Union[list, PIL.Image.Image, torch.Tensor]):
        """
        Preprocess the input images using the Hugging Face AutoImageProcessor.
        
        Args:
            images (Union[list, PIL.Image.Image, torch.Tensor]): Either a single PIL
            image, a list of PIL images, or a batch of images already as tensors.
        
        Returns:
            torch.Tensor: A batch of images ready for the model.
        """
        # If the input is a list (or a single image), process with the auto processor.
        if torch.is_tensor(images):
            return self.tensor_processor(images.to(torch.float32) / 255.0)
        elif isinstance(images, list):
            processed = self.processor(images=images, return_tensors="pt")
            return processed["pixel_values"]
        elif isinstance(images, Image.Image):
            processed = self.processor(images=images, return_tensors="pt")
            return processed["pixel_values"]
        
        raise ValueError("Unrecognized image type given for processing.")

    def forward(self, image_I: torch.Tensor, image_G: torch.Tensor):
        """
        Processes two images in one forward pass. The images are first concatenated
        and processed with the DINOv2 encoder. Their tokens are then split, transposed,
        and fed into a Transformer decoder, where tokens_I act as query and tokens_G
        as keys/values. The prediction head processes the CLS token (first token) from
        the decoder output.
        
        Args:
            image_I (torch.Tensor): Query image tensor; shape: [B, 3, H, W].
            image_G (torch.Tensor): Key/Value image tensor; shape: [B, 3, H, W].
        
        Returns:
            torch.Tensor: The final output prediction for each batch element.
        """
        B = image_I.shape[0]
        #TODO: Process in parallel
        tokens_I = self.dino(image_I)['last_hidden_state']
        tokens_G = self.dino(image_G)['last_hidden_state']

        # Permute dimensions to [seq_length, batch_size, embed_dim] for the decoder.
        tokens_I = tokens_I.transpose(0, 1)
        tokens_G = tokens_G.transpose(0, 1)

        # Pass through the Transformer decoder.
        decoder_out = self.decoder(tgt=tokens_I, memory=tokens_G)

        # Use the CLS token (assumed to be the first token) from the decoder output.
        cls_token = decoder_out[0]  # shape: [B, embed_dim]
        
        output = self.head(cls_token)
        return output

    def sim_ete(self, image_I: torch.Tensor, image_G: torch.Tensor) -> torch.Tensor:
        """
        Computes the distance between two images.
        
        Args:
            image_I (torch.Tensor): Query image tensor; shape: [B, 3, H, W].
            image_G (torch.Tensor): Key/Value image tensor; shape: [B, 3, H, W].
        
        Returns:
            torch.Tensor: A tensor of distance scores for each batch element.
        """
        goal_dist = self.forward(image_I, image_G)
        if self.dist_type == "linear":
            return -goal_dist
        elif self.dist_type == "l2":
            return -torch.linalg.norm(goal_dist, dim=-1)

    def sim(self, goal_dist: torch.Tensor) -> torch.Tensor:
        """
        Computes the distance between two images given the already encoded distance.
        
        Args:
            dense_dist (torch.Tensor): Encoded distance.
        
        Returns:
            torch.Tensor: A tensor of distance scores for each batch element.
        """
        if self.dist_type == "linear":
            return -goal_dist
        elif self.dist_type == "l2":
            return -torch.linalg.norm(goal_dist, dim=-1)


# Example usage:
if __name__ == "__main__":
    model = DistanceModel(dist_type="l2")
    batch_size = 2

    # Create dummy input images (assuming 3x224x224 tensors).
    image_I = torch.randn(batch_size, 3, 224, 224)
    image_G = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass through the model.
    output = model(image_I, image_G)
    print("Output shape:", output.shape)
    
    # Compute similarity between images.
    similarity_scores = model.sim_ete(image_I, image_G)
    print("Similarity scores:", similarity_scores)

    similarity_scores = model.sim(output)
    print("Similarity scores:", similarity_scores)
