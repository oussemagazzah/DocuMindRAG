"""initial schema

Revision ID: 001_initial
Revises:
Create Date: 2026-08-31
"""
from alembic import op
import sqlalchemy as sa


revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.Text(), nullable=False, server_default="processing"),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("doc_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("char_end", sa.Integer(), nullable=False, server_default="0"),
    )

    op.create_index("idx_chunks_doc_id", "chunks", ["doc_id"])
    op.create_index("idx_chunks_page", "chunks", ["doc_id", "page_number"])


def downgrade() -> None:
    op.drop_index("idx_chunks_page", "chunks")
    op.drop_index("idx_chunks_doc_id", "chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
