import torch.nn as nn



class SemanticSAM(nn.Module):
    def __init__(self,
                 image_encoder,
                 mask_decoder,
                 prompt_encoder,
                #  freeze_image_encoder=False,
                 ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        # # freeze prompt encoder
        # for param in self.prompt_encoder.parameters():
        #     param.requires_grad = False

        # self.freeze_image_encoder = freeze_image_encoder
        # if self.freeze_image_encoder:
        #     for param in self.image_encoder.parameters():
        #         param.requires_grad = False

    def forward(self, image):

        # do not compute gradients for pretrained prompt encoder

        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        # with torch.no_grad():
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )  # (B, 1, 256, 256)

        return low_res_masks