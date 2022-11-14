from dotenv import load_dotenv
import os
load_dotenv()
os.system(f"pg_dump --dbname=postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_DBHOST')}:5432/{os.getenv('POSTGRES_DBNAME')} > {os.getenv('NOMBRE_APELLIDO')}_sql_test.pgsql")