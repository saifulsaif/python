from mysql.connector import Error

class DatabaseOperations:
    """
    Database operations class for executing queries
    """
    def __init__(self, connection):
        self.connection = connection

    def fetch_all(self, query, params=None):
        """
        Fetch all records from database
        """
        try:
            cursor = self.connection.cursor(dictionary=True)
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            records = cursor.fetchall()
            cursor.close()
            return records
        except Error as e:
            print(f"Error while fetching data: {e}")
            return None

    def fetch_one(self, query, params=None):
        """
        Fetch single record from database
        """
        try:
            cursor = self.connection.cursor(dictionary=True)
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            record = cursor.fetchone()
            cursor.close()
            return record
        except Error as e:
            print(f"Error while fetching data: {e}")
            return None

    def execute_query(self, query, params=None):
        """
        Execute INSERT, UPDATE, DELETE queries
        """
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.connection.commit()
            affected_rows = cursor.rowcount
            cursor.close()
            print(f"{affected_rows} row(s) affected")
            return True
        except Error as e:
            print(f"Error while executing query: {e}")
            self.connection.rollback()
            return False

    def get_tables(self):
        """
        Get all tables in the database
        """
        query = "SHOW TABLES"
        return self.fetch_all(query)
