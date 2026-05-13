"""
Module cấu hình - đọc các cài đặt từ biến môi trường bằng pydantic-settings.
"""
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List


class Settings(BaseSettings):
    # --- Cấu hình MongoDB ---
    # MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_URI: str = "mongodb+srv://phamlong123np_db_user:f1YScvlLYLKMFQD0@cluster0.sdeftf0.mongodb.net/?appName=Cluster0"
    MONGODB_DB_NAME: str = "fresh-food"
    MONGODB_COLLECTION: str = "products"

    # --- Cấu hình Cache (Bộ nhớ đệm) ---
    # Thời gian sống của cache tính bằng phút
    CACHE_TTL_MINUTES: int = 30

    # --- Giới hạn dữ liệu ---
    # Số lượng sản phẩm tối đa lấy từ DB để tính toán AI
    MAX_PRODUCTS_FETCH: int = 10000

    # --- Cấu hình Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False

    # --- Cấu hình CORS (Cho phép truy cập từ domain khác) ---
    ALLOWED_ORIGINS: str = "*"

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v: str) -> str:
        """Xử lý giá trị đầu vào cho ALLOWED_ORIGINS."""
        return v

    def get_allowed_origins(self) -> List[str]:
        """Chuyển đổi chuỗi ALLOWED_ORIGINS thành danh sách các domain được phép."""
        if self.ALLOWED_ORIGINS.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

    model_config = {
        "env_file": ".env",           # Đọc file .env nếu có
        "env_file_encoding": "utf-8",
        "extra": "ignore"             # Bỏ qua các biến không định nghĩa trong class
    }


# Khởi tạo instance settings để sử dụng toàn bộ ứng dụng
settings = Settings()
