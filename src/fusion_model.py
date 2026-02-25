import torch.nn as nn
import torch

# class FusionModel(torch.nn.Module):
#     def __init__(self, input_dim, num_classes=3, hidden_dim=256, dropout=0.3):
#         super().__init__()
#         self.classifier = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(dropout),
#             torch.nn.Linear(hidden_dim, num_classes)
#         )

#     def forward(self, x):
#         return self.classifier(x)

class FusionModel(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=3,
        hidden_dim=256,
        dropout=0.3,
        use_second_layer=False,
        use_gating=False
    ):
        super().__init__()

        self.use_gating = use_gating

        # ---------- Gating mechanism ----------
        if use_gating:
            # self.gate = torch.nn.Sequential(
            #     torch.nn.Linear(input_dim, hidden_dim),
            #     torch.nn.Sigmoid()
            # )
            self.gate = torch.nn.Sequential(
                torch.nn.Linear(input_dim, input_dim),
                torch.nn.Sigmoid()
            )
        else:
            self.gate = None

        # ---------- Build fusion layers dynamically ----------
        layers = []

        # First fusion transformation (always exists)
        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))

        # Optional second fusion layer
        if use_second_layer:
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))

        # Output classifier
        layers.append(torch.nn.Linear(hidden_dim, num_classes))

        self.classifier = torch.nn.Sequential(*layers)

    def forward(self, x):
        # ---------- Apply gating if enabled ----------
        if self.gate is not None:
            gate_weight = self.gate(x)
            x = x * gate_weight

        return self.classifier(x)
