"""Alembic environment configuration for DocuMindRAG."""

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.config import get_settings

settings = get_settings()

if settings.database_url:
    config.set_main_option("sqlalchemy.url", settings.database_url)
else:
    config.set_main_option("sqlalchemy.url", f"sqlite:///{settings.database_path}")


def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=None, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
