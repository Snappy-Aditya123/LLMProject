import sqlite3
from typing import List, Tuple

class ChatDatabase:
    def __init__(self, db_name: str = "chatbot.db"):
        """
        Initialize the ChatDatabase class.
        :param db_name: Name of the SQLite database file.
        """
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self._create_tables()

    def _create_tables(self):
        """
        Create the necessary tables for storing chats.
        """
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS all_chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS last_three_chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def insert_chat(self, user_message: str, bot_response: str):
        """
        Insert a new chat into the all_chats table and update the last_three_chats table.
        :param user_message: The user's message.
        :param bot_response: The bot's response.
        """
        self.cursor.execute(
            "INSERT INTO all_chats (user_message, bot_response) VALUES (?, ?)",
            (user_message, bot_response)
        )
        self.connection.commit()
        self._update_last_three_chats(user_message, bot_response)

    def _update_last_three_chats(self, user_message: str, bot_response: str):
        """
        Update the last_three_chats table to keep only the last three chats.
        :param user_message: The user's message.
        :param bot_response: The bot's response.
        """
        self.cursor.execute(
            "INSERT INTO last_three_chats (user_message, bot_response) VALUES (?, ?)",
            (user_message, bot_response)
        )
        self.connection.commit()

        # Ensure only the last three chats are kept
        self.cursor.execute(
            """
            DELETE FROM last_three_chats
            WHERE id NOT IN (
                SELECT id FROM last_three_chats ORDER BY timestamp DESC LIMIT 3
            )
            """
        )
        self.connection.commit()

    def fetch_all_chats(self) -> List[Tuple[int, str, str, str]]:
        """
        Fetch all chats from the all_chats table.
        :return: List of tuples containing all chat records.
        """
        self.cursor.execute("SELECT * FROM all_chats ORDER BY timestamp ASC")
        return self.cursor.fetchall()

    def fetch_last_three_chats(self) -> List[Tuple[int, str, str, str]]:
        """
        Fetch the last three chats from the last_three_chats table.
        :return: List of tuples containing the last three chat records.
        """
        self.cursor.execute("SELECT * FROM last_three_chats ORDER BY timestamp DESC")
        return self.cursor.fetchall()

    def delete_chat_by_id(self, chat_id: int, table: str = "all_chats") -> bool:
        """
        Delete a chat by its ID from the specified table.
        :param chat_id: The ID of the chat to delete.
        :param table: The table from which to delete the chat (default is "all_chats").
        :return: True if a row was deleted, False otherwise.
        """
        if table not in ["all_chats", "last_three_chats"]:
            raise ValueError("Invalid table name. Use 'all_chats' or 'last_three_chats'.")

        self.cursor.execute(f"DELETE FROM {table} WHERE id = ?", (chat_id,))
        self.connection.commit()
        return self.cursor.rowcount > 0

    def close(self):
        """
        Close the database connection.
        """
        self.connection.close()



class JobInfoDB:
    def __init__(self, db_name="job_info.db"):
        self.db_name = db_name
        self.connection = sqlite3.connect(self.db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        """Create the jobs table if it doesn't exist."""
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                jobtitle TEXT NOT NULL,
                employername TEXT NOT NULL,
                location TEXT NOT NULL,
                mainsalary REAL,
                maxsalary REAL,
                jobdescription TEXT
            )
            """
        )
        self.connection.commit()

    def add_job(self, jobtitle, employername, location, mainsalary, maxsalary, jobdescription):
        """Add a new job record to the database."""
        self.cursor.execute(
            """
            INSERT INTO jobs (jobtitle, employername, location, mainsalary, maxsalary, jobdescription)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (jobtitle, employername, location, mainsalary, maxsalary, jobdescription)
        )
        self.connection.commit()

    def retrieve_jobs(self, filter_query=None):
        """Retrieve jobs from the database with an optional filter query."""
        query = "SELECT * FROM jobs"
        if filter_query:
            query += f" WHERE {filter_query}"
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def delete_job(self, job_id):
        """Delete a job record by ID."""
        self.cursor.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        self.connection.commit()

    def close(self):
        """Close the database connection."""
        self.connection.close()

import sqlite3

class ArticleInfoDB:
    def __init__(self, db_name="article_info.db"):
        self.db_name = db_name
        self.connection = sqlite3.connect(self.db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        """Create the articles table if it doesn't exist."""
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                published TEXT NOT NULL,
                url TEXT NOT NULL,
                content TEXT
            )
            """
        )
        self.connection.commit()

    def add_article(self, title, published, url, content):
        """Add a new article record to the database."""
        self.cursor.execute(
            """
            INSERT INTO articles (title, published, url, content)
            VALUES (?, ?, ?, ?)
            """,
            (title, published, url, content)
        )
        self.connection.commit()

    def retrieve_articles(self, filter_query=None):
        """Retrieve articles from the database with an optional filter query."""
        query = "SELECT * FROM articles"
        if filter_query:
            query += f" WHERE {filter_query}"
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def delete_article(self, article_id):
        """Delete an article record by ID."""
        self.cursor.execute("DELETE FROM articles WHERE id = ?", (article_id,))
        self.connection.commit()

    def close(self):
        """Close the database connection."""
        self.connection.close()

# Example Usage
if __name__ == "__main__":
    db = ArticleInfoDB()

    # Add articles
    db.add_article(
        "Breaking News: AI Advances",
        "2025-01-19",
        "https://example.com/ai-news",
        "AI is revolutionizing the tech industry."
    )
    db.add_article(
        "Tech Innovations",
        "2025-01-18",
        "https://example.com/tech-innovations",
        "New gadgets and software are being developed at an unprecedented pace."
    )

    # Retrieve articles
    articles = db.retrieve_articles()
    print("All Articles:")
    for article in articles:
        print(article)

    # Retrieve articles with a filter
    filtered_articles = db.retrieve_articles("published = '2025-01-19'")
    print("\nArticles Published on 2025-01-19:")
    for article in filtered_articles:
        print(article)

    # Delete an article by ID
    db.delete_article(1)  # Delete the article with ID 1

    # Close the database connection
    db.close()