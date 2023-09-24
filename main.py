import openai
from bs4 import BeautifulSoup
import requests


class BookRecommendationSystem:
    def __init__(self):
        self.book_data = self.scrape_books_data()
        self.user_preferences = self.get_user_preferences()
        self.analyzed_books = self.analyze_books()
        self.recommendations = self.generate_recommendations()
        self.ai_generated_recommendations = self.integrate_ai_models()

    def scrape_books_data(self):
        url = "https://www.examplebookstore.com/books"
        book_data = []

        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            books = soup.find_all("div", class_="book")

            for book in books:
                title = book.find("h3", class_="title").text.strip()
                author = book.find("p", class_="author").text.strip()
                genre = book.find("p", class_="genre").text.strip()
                summary = book.find("div", class_="summary").text.strip()
                rating = float(book.find("span", class_="rating").text.strip())

                book_info = {
                    "title": title,
                    "author": author,
                    "genre": genre,
                    "summary": summary,
                    "rating": rating,
                }

                book_data.append(book_info)

        except requests.exceptions.RequestException as e:
            print(f"Error occurred while scraping book data: {e}")

        return book_data

    def get_user_preferences(self):
        user_preferences = {}

        favorite_books = ["To Kill a Mockingbird", "Pride and Prejudice", "1984"]
        favorite_genres = ["Mystery", "Fantasy"]
        interests = "I enjoy exploring complex characters and intricate plotlines."

        user_preferences["favorite_books"] = favorite_books
        user_preferences["favorite_genres"] = favorite_genres
        user_preferences["interests"] = interests

        return user_preferences

    def analyze_books(self):
        analyzed_books = []

        for book in self.book_data:
            sentiment = self.analyze_sentiment(book["summary"])
            book["sentiment"] = sentiment
            analyzed_books.append(book)

        return analyzed_books

    def analyze_sentiment(self, text):
        response = openai.Answer.create(
            search_model="davinci",
            model="davinci",
            question=text,
            documents=self.book_data,
            examples_context=text,
            max_documents=1,
            stop=None,
        )
        return response.choices[0].text.strip()

    def generate_recommendations(self):
        recommendations = []

        favorite_books = set(self.user_preferences["favorite_books"])
        favorite_genres = set(self.user_preferences["favorite_genres"])

        for book in self.analyzed_books:
            if (book["title"] not in favorite_books) and (
                book["genre"] in favorite_genres
            ):
                recommendations.append(book)

        return recommendations

    def integrate_ai_models(self):
        ai_generated_recommendations = []

        prompt = "Based on your reading interests and preferences, I would like to recommend the following books:\n\n"
        for recommendation in self.recommendations:
            prompt += f"{recommendation['title']} - {recommendation['author']}\nSummary: {recommendation['summary']}\n\n"

        response = openai.Completion.create(
            model="davinci",
            prompt=prompt,
            max_tokens=50,
            temperature=0.5,
            n=3,
            stop="\n",
        )

        for choice in response.choices:
            book_title = choice.text.split(" - ")[0]
            book_info = {"title": book_title.strip()}
            ai_generated_recommendations.append(book_info)

        return ai_generated_recommendations

    def display_recommendations(self):
        if self.ai_generated_recommendations:
            print("Here are some book recommendations based on your preferences:")
            for i, recommendation in enumerate(self.ai_generated_recommendations):
                print(f"{i+1}. {recommendation['title']}")
        else:
            print("No book recommendations available.")

    def run(self):
        self.display_recommendations()


if __name__ == "__main__":
    book_recommendation_system = BookRecommendationSystem()
    book_recommendation_system.run()
