# middleware.py
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from main import app as flask_app
from Home import streamlit_app

# Combine both apps
application = DispatcherMiddleware(
    flask_app,  # Flask app handles all routes except /streamlit
    {
        '/streamlit': streamlit_app  # Streamlit app mounted at /streamlit
    }
)

if __name__ == '__main__':
    run_simple('localhost', 8080, application,
               use_reloader=True, use_debugger=True)