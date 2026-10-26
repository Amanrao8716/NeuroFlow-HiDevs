from urllib.parse import quote

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = "NeuroFlow"
    app_env: str = "development"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "neuroflow"
    postgres_user: str = "neuroflow_app"
    postgres_password: str

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # OpenTelemetry
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"

    # Internal observability API
    observability_service_token: str | None = None
    neuroflow_api_url: str = "http://localhost:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def postgres_dsn(self) -> str:
        """Build the PostgreSQL connection string."""
        user = quote(self.postgres_user, safe="")
        password = quote(self.postgres_password, safe="")

        return (
            f"postgresql://{user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}"
            f"/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Build the Redis connection URL."""
        password = quote(self.redis_password, safe="")
        return f"redis://:{password}@{self.redis_host}:{self.redis_port}/0"


settings = Settings()