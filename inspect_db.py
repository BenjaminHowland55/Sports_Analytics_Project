"""Print the tables in nfl_analytics, then the columns and first 3 rows of the defensive-efficiency table.

Nothing about the schema is assumed: the table list and columns come from the database itself,
and the defensive-efficiency table is whichever table has an "epa_per_play..." column.

Run from this folder:  python inspect_db.py
"""
from sqlalchemy import inspect, text

from load_to_database import DB_NAME, connect

engine = connect(DB_NAME)
schema = inspect(engine)

tables = schema.get_table_names()
print(f"Tables in {DB_NAME}:")
for table in tables:
    print(f"  - {table}")

efficiency_tables = [t for t in tables
                     if any("epa_per_play" in col["name"] for col in schema.get_columns(t))]
if not efficiency_tables:
    raise SystemExit("\nNo table has an 'epa_per_play' column.")

for table in efficiency_tables:
    print(f"\nDefensive-efficiency table: {table}")
    print("Columns:")
    for col in schema.get_columns(table):
        print(f"  - {col['name']} ({col['type']})")

    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT * FROM `{table}` LIMIT 3"))
        names = list(result.keys())
        print("\nFirst 3 rows:")
        for row in result:
            print("  " + ", ".join(f"{name}={value}" for name, value in zip(names, row)))
