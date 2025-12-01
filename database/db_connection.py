import mysql.connector
from mysql.connector import Error

class DatabaseConnection:
    """
    Database connection class for MySQL database
    """
    def __init__(self):
        self.host = '127.0.0.1'
        self.port = 3306
        self.database = 'jomaghar'
        self.user = 'root'
        self.password = 'root'
        self.connection = None

    def connect(self):
        """
        Create database connection
        """
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )

            if self.connection.is_connected():
                db_info = self.connection.get_server_info()
                print(f"Successfully connected to MySQL Server version {db_info}")
                cursor = self.connection.cursor()
                cursor.execute("SELECT DATABASE();")
                record = cursor.fetchone()
                print(f"Connected to database: {record[0]}")
                cursor.close()
                return self.connection
        except Error as e:
            print(f"Error while connecting to MySQL: {e}")
            return None

    def disconnect(self):
        """
        Close database connection
        """
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("MySQL connection is closed")

    def get_connection(self):
        """
        Get the current connection
        """
        return self.connection
