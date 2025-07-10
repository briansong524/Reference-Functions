 

### Connecting to DB for data

def connect_db():

    pgcreds=json.loads(os.getenv("pgcreds"))

    postgres_connection_string = ''    

    for param in ['host','user','dbname','password','sslmode']:

        postgres_connection_string += param + "=" + pgcreds[param] + ' '

 

    postgresql_pool = psycopg2.pool.SimpleConnectionPool(1, 20, postgres_connection_string)

    if (postgresql_pool):

        print("Connection pool created successfully")

 

    # Get a connection from the connection pool

    conn = postgresql_pool.getconn()

    cursor = conn.cursor()

   

    return conn,cursor

 

def get_all_customers():

    conn, cur = connect_db()

    with conn.cursor() as cursor:

        cursor.execute(f"""

            SELECT DISTINCT(business_name)

            FROM project_echo.customer_interactions

        """)

        customers = cursor.fetchall()

    return [customer[0] for customer in customers]   # Extract text from results
