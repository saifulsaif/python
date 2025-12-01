import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import database modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_connection import DatabaseConnection
from database.db_operations import DatabaseOperations


def display_top_users_deposits():
    """
    Display top 10 users with highest total deposits using bar chart
    """
    # Create database connection
    db = DatabaseConnection()
    connection = db.connect()

    if connection:
        # Create database operations instance
        db_ops = DatabaseOperations(connection)

        # Fetch top 10 users with total deposit amounts
        query = """
            SELECT
                users.first_name,
                users.last_name,
                users.username,
                users.email,
                users.phone,
                SUM(user_deposits.amount) as total_amount
            FROM users
            INNER JOIN user_deposits ON users.id = user_deposits.user_id
            GROUP BY users.id, users.first_name, users.last_name, users.username, users.email, users.phone
            ORDER BY total_amount DESC
            LIMIT 10
        """

        users = db_ops.fetch_all(query)

        if users and len(users) > 0:
            print(f"\nTop {len(users)} Users by Total Deposits\n")

            # Prepare data for bar chart
            user_labels = []
            amounts = []

            for user in users:
                # Create label with username or name
                label = user['username'] if user['username'] else f"{user['first_name']} {user['last_name']}"
                user_labels.append(label)
                amounts.append(float(user['total_amount']))

                # Print user details
                print(f"User: {label}")
                print(f"  Email: {user['email']}")
                print(f"  Phone: {user['phone']}")
                print(f"  Total Deposits: ${user['total_amount']:,.2f}\n")

            # Create bar chart
            fig, ax = plt.subplots(figsize=(14, 8))

            # Create bars
            bars = ax.bar(user_labels, amounts, color='#4CAF50', alpha=0.8, edgecolor='black', linewidth=1.2)

            # Customize the chart
            ax.set_xlabel('Users', fontsize=12, fontweight='bold')
            ax.set_ylabel('Total Deposit Amount ($)', fontsize=12, fontweight='bold')
            ax.set_title('Top 10 Users by Total Deposit Amount', fontsize=16, fontweight='bold', pad=20)

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:,.2f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

            # Add grid for better readability
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)

            # Format y-axis to show currency
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

            # Adjust layout to prevent label cutoff
            plt.tight_layout()

            # Show the plot
            plt.show()

            print("Bar chart displayed successfully!")
        else:
            print("No deposit data found in the database")

        # Close the connection
        db.disconnect()
    else:
        print("Failed to connect to database")


if __name__ == "__main__":
    display_top_users_deposits()
