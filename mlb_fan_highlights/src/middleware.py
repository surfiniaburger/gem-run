# mlb_fan_highlights/src/middleware.py
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from main import app as flask_app


# Combine both apps
application = DispatcherMiddleware(
    flask_app, 

)

if __name__ == '__main__':
    run_simple('localhost', 8080, application,
               use_reloader=True, use_debugger=True)