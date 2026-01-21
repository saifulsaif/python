import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path to import database modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_connection import DatabaseConnection
from database.db_operations import DatabaseOperations


def display_all_users():
    """
    Display all users from the users table using matplotlib
    """
    # Create database connection
    db = DatabaseConnection()
    connection = db.connect()

    if connection:
        # Create database operations instance
        db_ops = DatabaseOperations(connection)

        # Fetch all users
        query = "SELECT first_name,last_name,username,email,phone FROM users"
        users = db_ops.fetch_all(query)

        if users and len(users) > 0:
            print(f"\nTotal Users: {len(users)}")

            # Get column names from first record
            columns = list(users[0].keys())

            # Prepare data for table display
            cell_text = []
            for user in users:
                row = [str(user[col])[:20] for col in columns]  # Limit text length
                cell_text.append(row)

            # Create figure and axis
            _, ax = plt.subplots(figsize=(12, 8))
            ax.axis('tight')
            ax.axis('off')

            # Create table
            table = ax.table(
                cellText=cell_text,
                colLabels=[col.upper() for col in columns],
                cellLoc='left',
                loc='center',
                colWidths=[0.15] * len(columns)
            )

            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)

            # Style header row
            for i in range(len(columns)):
                cell = table[(0, i)]
                cell.set_facecolor("#3D87E2")
                cell.set_text_props(weight='bold', color='white')

            # Alternate row colors
            for i in range(1, len(cell_text) + 1):
                for j in range(len(columns)):
                    cell = table[(i, j)]
                    if i % 2 == 0:
                        cell.set_facecolor('#f0f0f0')
                    else:
                        cell.set_facecolor('#ffffff')

            # Set title

            # Show the plot
            plt.tight_layout()
            plt.show()

            print("Table displayed successfully!")
        else:
            print("No users found in the database")

        # Close the connection
        db.disconnect()
    else:
        print("Failed to connect to database")


if __name__ == "__main__":
    display_all_users()
