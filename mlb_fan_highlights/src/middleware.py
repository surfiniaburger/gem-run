# mlb_fan_highlights/src/middleware.py
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.wrappers import Response
from main import app as flask_app
import streamlit.web.bootstrap as bootstrap
from streamlit.web.server.server import Server
import os

def streamlit_wsgi_app(environ, start_response):
    bootstrap.run(
        file=os.path.join(os.path.dirname(__file__), "Home.py"),
        command_line=[],
        args=[],
        flag_options={},
    )
    return Response('')(environ, start_response)

# Combine both apps
application = DispatcherMiddleware(
    flask_app,  # Flask app handles all routes except /streamlit
    {
        '/streamlit': streamlit_wsgi_app  # Streamlit app mounted at /streamlit with proper WSGI wrapper
    }
)

# Add error handling
def application_with_error_handling(environ, start_response):
    try:
        return application(environ, start_response)
    except Exception as e:
        # Log the error
        print(f"Error in application: {str(e)}")
        # Return a 500 error response
        status = '500 Internal Server Error'
        response_headers = [('Content-type', 'text/plain')]
        start_response(status, response_headers)
        return [b'Internal Server Error']

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 8080, application_with_error_handling,
               use_reloader=True, use_debugger=True)