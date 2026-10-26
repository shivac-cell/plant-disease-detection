"""
Plant Disease Detection and Smart Pesticide & Fertilizer Recommendation System
Flask Application Entry Point and Application Factory (Phases 1 & 2)
"""

import os
from typing import Optional, Type

from flask import Flask
from flask_wtf.csrf import CSRFProtect

from config import Config, get_config
from routes.auth_routes import auth_bp
from routes.main_routes import register_routes
from translations.i18n import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    get_translation,
    load_catalogs,
)

# Global CSRF protection instance
csrf = CSRFProtect()


def create_app(config_class: Optional[Type[Config]] = None) -> Flask:
    """
    Application factory pattern for Flask.
    Initializes Flask app, registers configurations, attaches CSRF protection,
    loads translation catalogs, registers i18n context processors, and registers application routes.
    """
    frontend_dir = Config.BASE_DIR / "frontend"
    app = Flask(
        __name__,
        template_folder=str(frontend_dir / "templates"),
        static_folder=str(frontend_dir / "static"),
    )

    # Load configuration
    if config_class is None:
        config_class = get_config()
    app.config.from_object(config_class)

    # Initialize CSRF protection
    csrf.init_app(app)

    # Ensure upload directory exists
    upload_folder = app.config.get("UPLOAD_FOLDER", Config.UPLOAD_FOLDER)
    upload_folder.mkdir(parents=True, exist_ok=True)

    # Initialize Internationalization Catalogs
    load_catalogs()
    from translations.i18n import translate_crop, translate_disease

    # Session & Language setup
    from flask import session

    @app.before_request
    def ensure_language():
        if "lang" not in session or session["lang"] not in SUPPORTED_LANGUAGES:
            session["lang"] = DEFAULT_LANGUAGE

    @app.context_processor
    def inject_i18n():
        current_lang = session.get("lang", DEFAULT_LANGUAGE)
        if current_lang not in SUPPORTED_LANGUAGES:
            current_lang = DEFAULT_LANGUAGE
        return {
            "t": lambda key, **kwargs: get_translation(key, lang=session.get("lang", DEFAULT_LANGUAGE), **kwargs),
            "t_crop": lambda crop: translate_crop(crop, lang=session.get("lang", DEFAULT_LANGUAGE)),
            "t_disease": lambda disease: translate_disease(disease, lang=session.get("lang", DEFAULT_LANGUAGE)),
            "current_lang": current_lang,
            "supported_languages": SUPPORTED_LANGUAGES,
        }

    # Register application blueprints and routes
    app.register_blueprint(auth_bp)
    register_routes(app)

    return app


# Application instance for standard WSGI servers or flask CLI
app = create_app()

if __name__ == "__main__":
    # Run development server
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=app.config.get("DEBUG", False))
