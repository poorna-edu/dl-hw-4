from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Normalize input
        self.register_buffer('input_mean', torch.tensor(INPUT_MEAN[:2], dtype=torch.float32))
        self.register_buffer('input_std', torch.tensor(INPUT_STD[:2], dtype=torch.float32))

        hidden_dim = 128
        num_layers = 5

        input_dim = n_track * 4
        output_dim = n_waypoints * 2

        # Define layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input using predefined mean and std
        """
        return (x - self.input_mean.to(x.device)[None, None, :]) / self.input_std.to(x.device)[None, None, :]

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size, n_track, _ = track_left.shape

        # Normalize inputs
        track_left = self.normalize_input(track_left)
        track_right = self.normalize_input(track_right)

        # New shape: (b, n_track, 4)
        track_features = torch.cat([track_left, track_right], dim=-1)

        # New shape: (b, n_track * 4)
        track_features = track_features.reshape(batch_size, -1)

        # Assuming self.model outputs a tensor of shape (b, n_waypoints * 2)
        waypoints_flat = self.model(track_features)

        # Final shape: (b, n_waypoints, 2)
        n_waypoints = waypoints_flat.shape[1] // 2
        waypoints = waypoints_flat.reshape(batch_size, n_waypoints, 2)

        return waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Track points into a latent space
        self.track_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Transformer decoder configuration
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=4, 
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )

        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.size(0)

        # Shape: (b, 2 * n_track, 2)
        track_points = torch.cat([track_left, track_right], dim=1)

        # Shape: (b, 2 * n_track, d_model)
        track_encoded = self.track_encoder(track_points)

        # Shape: (n_waypoints, d_model) -> (b, n_waypoints, d_model)
        queries = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # Shape after transform: (b, n_waypoints, d_model)
        transformer_output = self.transformer(
            src=track_encoded, tgt=queries
        )

        # Shape: (b, n_waypoints, 2)
        waypoints = self.output_proj(transformer_output)

        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        n_blocks = 4
        in_channels = 32 

        # Convolutional layers
        cnn_layers = [
            torch.nn.Conv2d(3, in_channels, kernel_size=11, stride=2, padding=5),
            torch.nn.ReLU(),
        ]

        # Add convolutional layers with increasing channels
        c1 = in_channels
        for _ in range(n_blocks):
            c2 = c1 * 2
            cnn_layers.extend([
                torch.nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
                torch.nn.BatchNorm2d(c2),
                torch.nn.ReLU(),
            ])
            c1 = c2

        # Final convolutional 
        cnn_layers.append(torch.nn.Conv2d(c1, c1, kernel_size=1))
        cnn_layers.append(torch.nn.AdaptiveAvgPool2d(1))

        self.network = torch.nn.Sequential(*cnn_layers)

        # Fully connected layers for final prediction
        self.fcc = nn.Sequential(
            nn.Linear(c1, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_waypoints * 2),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Pass through CNN layers
        z = self.network(x)
        z = z.view(z.size(0), -1)
        logits = self.fcc(z)
        waypoints = logits.view(-1, self.n_waypoints, 2)
        return waypoints


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
