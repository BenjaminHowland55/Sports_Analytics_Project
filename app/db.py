import os
from pathlib import Path

import certifi
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

# Locally, copy settings from .env into environment variables. On Vercel there is no .env file;
# DATABASE_PASSWORD is set in the project's Environment Variables settings instead.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

url = URL.create(
    "mysql+pymysql",
    username="2sjjjYZZAfxKqVv.root",
    password=os.environ["DATABASE_PASSWORD"],
    host="gateway01.us-west-2.prod.aws.tidbcloud.com",
    port=4000,
    database="nfl_analytics",
)

# TiDB Cloud requires an encrypted (TLS) connection
engine = create_engine(url, connect_args={"ssl": {"ca": certifi.where()}}, pool_pre_ping=True)
