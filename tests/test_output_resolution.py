"""Tests for output_resolution support (Nano Banana Pro / Nano Banana 2)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from PIL import Image as PILImage
from pydantic import ValidationError

from paperbanana.core.config import Settings
from paperbanana.providers.image_gen.google_imagen import GoogleImagenGen


# ── Config validation ────────────────────────────────────────────────


class TestOutputResolutionConfig:
    """output_resolution config validation."""

    def test_default_is_2k(self):
        settings = Settings()
        assert settings.output_resolution == "2K"

    @pytest.mark.parametrize("value", ["1K", "2K", "4K", "512"])
    def test_valid_values_accepted(self, value):
        settings = Settings(output_resolution=value)
        assert settings.output_resolution == value

    @pytest.mark.parametrize("input_val,expected", [
        ("1k", "1K"),
        ("2k", "2K"),
        ("4k", "4K"),
        ("  2K  ", "2K"),
    ])
    def test_case_normalization(self, input_val, expected):
        settings = Settings(output_resolution=input_val)
        assert settings.output_resolution == expected

    @pytest.mark.parametrize("bad", ["3K", "8K", "HD", "full", ""])
    def test_invalid_values_rejected(self, bad):
        with pytest.raises(ValidationError, match="output_resolution must be one of"):
            Settings(output_resolution=bad)

    def test_from_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump({"pipeline": {"output_resolution": "4K"}}, f)
            path = f.name
        try:
            settings = Settings.from_yaml(path)
            assert settings.output_resolution == "4K"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_from_yaml_lowercase_normalized(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump({"pipeline": {"output_resolution": "4k"}}, f)
            path = f.name
        try:
            settings = Settings.from_yaml(path)
            assert settings.output_resolution == "4K"
        finally:
            Path(path).unlink(missing_ok=True)


# ── Model selection ──────────────────────────────────────────────────


class TestModelSelection:
    """Nano Banana Pro vs Nano Banana 2 model selection."""

    def test_nano_banana_pro_default(self):
        settings = Settings(image_provider="google_imagen", google_api_key="k")
        assert settings.image_model == "gemini-3-pro-image-preview"

    def test_nano_banana_2_via_image_model(self):
        settings = Settings(
            image_provider="google_imagen",
            image_model="gemini-3.1-flash-image-preview",
            google_api_key="k",
        )
        assert settings.image_model == "gemini-3.1-flash-image-preview"

    def test_nano_banana_2_via_google_image_model_override(self):
        settings = Settings(
            image_provider="google_imagen",
            google_image_model="gemini-3.1-flash-image-preview",
            google_api_key="k",
        )
        assert settings.effective_image_model == "gemini-3.1-flash-image-preview"


# ── GoogleImagenGen resolution handling ──────────────────────────────


class TestGoogleImagenResolution:
    """GoogleImagenGen output_resolution → image_size wiring."""

    def _make_provider(self, model="gemini-3-pro-image-preview"):
        return GoogleImagenGen(api_key="test-key", model=model)

    def test_validate_resolution_pro_accepts_1k_2k_4k(self):
        gen = self._make_provider("gemini-3-pro-image-preview")
        assert gen._validate_resolution("1K") == "1K"
        assert gen._validate_resolution("2K") == "2K"
        assert gen._validate_resolution("4K") == "4K"

    def test_validate_resolution_pro_rejects_512(self):
        gen = self._make_provider("gemini-3-pro-image-preview")
        assert gen._validate_resolution("512") == "1K"  # falls back

    def test_validate_resolution_nano_banana_2_accepts_512(self):
        gen = self._make_provider("gemini-3.1-flash-image-preview")
        assert gen._validate_resolution("512") == "512"

    def test_validate_resolution_nano_banana_2_accepts_all(self):
        gen = self._make_provider("gemini-3.1-flash-image-preview")
        for res in ("512", "1K", "2K", "4K"):
            assert gen._validate_resolution(res) == res

    @pytest.mark.asyncio
    async def test_generate_passes_output_resolution_to_image_config(self):
        """output_resolution overrides the dimension-based image_size."""
        gen = self._make_provider("gemini-3-pro-image-preview")

        mock_client = MagicMock()
        mock_part = MagicMock()
        mock_part.as_image.return_value = PILImage.new("RGB", (100, 100))
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [mock_part]
        mock_client.models.generate_content.return_value = mock_response
        gen._client = mock_client

        await gen.generate(
            prompt="test",
            width=1024,
            height=1024,
            output_resolution="4K",
        )

        call_kwargs = mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config.image_config.image_size == "4K"

    @pytest.mark.asyncio
    async def test_generate_without_resolution_uses_dimension_based(self):
        """Without output_resolution, image_size comes from dimensions."""
        gen = self._make_provider()

        mock_client = MagicMock()
        mock_part = MagicMock()
        mock_part.as_image.return_value = PILImage.new("RGB", (100, 100))
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [mock_part]
        mock_client.models.generate_content.return_value = mock_response
        gen._client = mock_client

        await gen.generate(
            prompt="test",
            width=1024,
            height=1024,
        )

        call_kwargs = mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        # 1024px → "1K"
        assert config.image_config.image_size == "1K"


# ── Visualizer agent wiring ──────────────────────────────────────────


class TestVisualizerResolution:
    """VisualizerAgent passes output_resolution to image_gen."""

    @pytest.mark.asyncio
    async def test_diagram_passes_resolution(self, tmp_path):
        from paperbanana.agents.visualizer import VisualizerAgent

        mock_image_gen = AsyncMock()
        mock_image_gen.generate.return_value = PILImage.new("RGB", (100, 100))

        mock_vlm = AsyncMock()

        visualizer = VisualizerAgent(
            image_gen=mock_image_gen,
            vlm_provider=mock_vlm,
            prompt_dir="prompts",
            output_dir=str(tmp_path),
        )

        # Mock load_prompt to avoid file-not-found
        visualizer.load_prompt = MagicMock(return_value="Generate: {description}")

        await visualizer.run(
            description="test diagram",
            output_resolution="4K",
        )

        mock_image_gen.generate.assert_called_once()
        call_kwargs = mock_image_gen.generate.call_args.kwargs
        assert call_kwargs["output_resolution"] == "4K"

    @pytest.mark.asyncio
    async def test_diagram_none_resolution(self, tmp_path):
        from paperbanana.agents.visualizer import VisualizerAgent

        mock_image_gen = AsyncMock()
        mock_image_gen.generate.return_value = PILImage.new("RGB", (100, 100))

        mock_vlm = AsyncMock()

        visualizer = VisualizerAgent(
            image_gen=mock_image_gen,
            vlm_provider=mock_vlm,
            prompt_dir="prompts",
            output_dir=str(tmp_path),
        )
        visualizer.load_prompt = MagicMock(return_value="Generate: {description}")

        await visualizer.run(
            description="test diagram",
        )

        call_kwargs = mock_image_gen.generate.call_args.kwargs
        assert call_kwargs["output_resolution"] is None


# ── Pipeline wiring ──────────────────────────────────────────────────


class TestPipelineResolution:
    """Pipeline passes settings.output_resolution to visualizer."""

    def test_settings_output_resolution_in_pipeline(self):
        """Settings.output_resolution is accessible for pipeline use."""
        settings = Settings(output_resolution="4K")
        assert settings.output_resolution == "4K"

        settings2 = Settings(output_resolution="512")
        assert settings2.output_resolution == "512"
