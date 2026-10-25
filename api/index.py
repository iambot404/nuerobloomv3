import sys
import os

# Ensure project root directory is in Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from urllib.parse import parse_qs, urlencode
from werkzeug.middleware.proxy_fix import ProxyFix
from app import app


# WSGI Middleware to correct PATH_INFO when Vercel serverless rewrites route requests
class VercelPathFix:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path_info = environ.get('PATH_INFO', '')

        # 1. Check if Vercel passed original path via __vercel_route query param
        qs = environ.get('QUERY_STRING', '')
        if '__vercel_route' in qs:
            params = parse_qs(qs, keep_blank_values=True)
            route_vals = params.pop('__vercel_route', None)
            if route_vals and route_vals[0] and route_vals[0] not in (':path*', '/:path*'):
                route = route_vals[0]
                if not route.startswith('/'):
                    route = '/' + route
                if len(route) > 1 and route.endswith('/'):
                    route = route.rstrip('/')
            else:
                route = '/'
            # Reconstruct clean QUERY_STRING without internal param
            environ['QUERY_STRING'] = urlencode(params, doseq=True)
            environ['PATH_INFO'] = route
            return self.wsgi_app(environ, start_response)

        # 2. Check if Vercel proxy headers passed original request path
        forwarded = (
            environ.get('HTTP_X_FORWARDED_URI') or
            environ.get('RAW_URI') or
            environ.get('REQUEST_URI')
        )
        if forwarded and forwarded not in ('/api/index', '/api/index.py', '/api', '/api/'):
            clean_path = forwarded.split('?')[0]
            environ['PATH_INFO'] = clean_path if clean_path else '/'
        elif path_info in ('/api/index', '/api/index.py', '/api', '/api/'):
            environ['PATH_INFO'] = '/'
        elif path_info.startswith('/api/index/'):
            environ['PATH_INFO'] = path_info[len('/api/index'):] or '/'
        elif path_info.startswith('/api/index.py/'):
            environ['PATH_INFO'] = path_info[len('/api/index.py'):] or '/'

        return self.wsgi_app(environ, start_response)


# Apply ProxyFix for correct protocol and host headers behind Vercel reverse proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)

# Apply VercelPathFix to handle rewritten URL routing
app.wsgi_app = VercelPathFix(app.wsgi_app)

# Vercel serverless function entrypoint
if __name__ == '__main__':
    app.run()
