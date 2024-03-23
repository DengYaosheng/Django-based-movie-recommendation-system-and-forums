# Django-based-movie-recommendation-system-and-forums
Internship in Kyoto Univ Graduate School of Informatics - (nonconfidential)
# Graduation Project - Movie Recommendation System and Forum Based on Django

## Description

1. It is recommended for beginners to use PyCharm. You can download the professional version and try it free for 30 days: [Link](https://www.jetbrains.com/pycharm/)
2. Register regular users through the web interface and create administrators using `createsuperuser` command. Detailed commands are provided below.
3. Import movie information using `insert_movies_script.py` (This will delete all existing information!)
4. Frontend displays include functions such as most browsed, highest rated, most favorited, etc. You can make them sound more indirect, like "hottest movies" or "popular rankings". Each category displays 10 items.

I guess you prefer user-based recommendation for your system, while item recommendation is based on projects. Both approaches are explained below.

## Technologies Used

Frontend: Bootstrap3 CSS framework
Backend: Django 2.2.1 + SQLite3 database (MVC framework)
Data: Python asynchronous crawler to fetch data from Douban Top 250 and save it to local CSV files
Main Features: Inputting movie information, user ratings, movie tag classification, movie recommendation, movie sharing, movie favoriting, and backend management system.
Adopts MVC architecture, frontend pages are implemented using Django template, facilitating template reuse. The organization of frontend pages is clear.

## Recommendation Algorithm

Calculate the distance between users through collaborative filtering and other users, then filter. If the number of users is insufficient and the number of recommendations is less than 15, automatically fill in a portion from all unrated movies in descending order of views.

### User-based Recommendation

1. Users need to rate movies. Calculate similarity based on the rated portion by users. If the user has not rated or there are no other users, return in descending order of views.
2. Use Pearson algorithm to calculate the distance between users, find the N nearest users, and return the rated movies among these users (and unseen by the user to be recommended).

### Item-based Recommendation

1. Calculate the similarity matrix of items.
2. Traverse the items already rated by the current user, calculate the similarity distance with unrated items.
3. Sort by similarity distance and return.

## Key Implemented Features

1. Login/register page
2. Movie classification, sorting, searching, rating, and sorting based on collaborative filtering.
3. Weekly and monthly recommendations based on collaborative filtering.
4. Features such as movie sharing events and user registration for activities (to be added separately)
5. Forum functionality for posting messages (to be added separately)
6. ALS algorithm based on Spark (to be added separately)
7. MySQL adaptation
8. Integration with MovieLens dataset

## References

[Recommendation Algorithm - Collaborative Filtering - JianShu](https://www.jianshu.com/p/5463ab162a58)
[What's the Difference Between Collaborative Filtering and Content-Based Recommendation? - Zhihu](https://www.zhihu.com/question/19971859)

## Fixes

1. Incorrect homepage navigation links
2. Empty homepage
3. Login/register page
4. Recommendation redirects to login
5. Weekly recommendations randomly when users haven't rated
6. Sorting by number of favorites
7. Redesigned action and UserAction model, separating UserAction

## Movie Model

1. Views: Number of views each time the page is refreshed
2. Favorites: Many-to-many field for users, each user can favorite once
3. Rating: Each user rates once
4. Like function for comments under movies

## Installation and Running

## Dependencies Installation

1. Import the project into PyCharm and configure the Python interpreter (Python 3.7 or below). You can install via Conda or other virtual environments.
2. Open the terminal and run `pip install -r requirements.txt`. If pip is not found, download `get-pip.py` and run `python get-pip.py`.
3. During pip installation, if encountering C++ 14 dependency issues, install the C++ dependency tool. If you can't find it, ask me. If installation is slow, change to a mirror in China.
4. Once installation is successful, proceed to the running phase.

## Running

1. Run the server: `python manage.py runserver`
2. If there is no data, run the data migration script starting with "populate" in the project root directory.
3. Create a superuser: `python manage.py createsuperuser` (Password input will not be visible in the terminal)
4. Access the admin panel: [127.0.0.1:8000/admin](http://127.0.0.1:8000/admin)

For permanent updates and maintenance support, please contact me.
For other issues, please contact me.

## Functionality of Each File

1. `media/`: Directory for storing static files, such as images.
2. `movie/`: Default app in Django, responsible for settings configuration, URL routing, deployment, etc.
3. `static/`: Directory for storing CSS and JS files.
4. `user/`: Main app, most of the program's code resides here. `user/migrations` contains auto-generated database migration files, `user/templates` contains frontend template files, `user/admins.py` contains admin backend code, `user/forms.py` contains frontend form code, `user/models.py` contains database ORM models, `user/serializers.py` contains RESTful files (not relevant), `user/urls` registers the routes, and `user/views` handles frontend requests and interacts with the backend database (i.e., controller module).
5. `cache_keys.py`: File for storing cache key names (ignore).
6. `db.sqlite3`: Database file.
7. `douban_crawler.py`: Douban crawler file.
8. `manage.py`: Main program for running, start from here.
9. `populate_movies_script.py`: Fills movie data into the database.
10. `populate_user_rate.py`: Randomly generates user ratings.
