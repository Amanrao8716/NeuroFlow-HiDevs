\getenv app_user POSTGRES_APP_USER
\getenv app_password POSTGRES_APP_PASSWORD

SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'app_user', :'app_password')
\gexec

SELECT format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), :'app_user')
\gexec
GRANT USAGE, CREATE ON SCHEMA public TO :"app_user";
