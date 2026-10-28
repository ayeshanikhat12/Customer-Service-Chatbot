"""Inference and response mapping for the chatbot."""
from functools import lru_cache
from pathlib import Path
import sys

import joblib

try:
    from .preprocessing import clean_text
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "chatbot_model.pkl"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.pkl"

RESPONSES = {
    "cancel_order": "I can guide you through cancelling an order. Please check the order status in your account; orders already shipped may not be cancellable.",
    "change_order": "To change an order, open your order details and check whether editing is still available. Changes may be limited after processing begins.",
    "change_shipping_address": "You can usually update the shipping address from your order details before dispatch. If it has shipped, contact customer support promptly.",
    "check_cancellation_fee": "Cancellation fees depend on the order and its status. Please review the cancellation terms shown for your order or contact support for confirmation.",
    "check_invoice": "You can usually find invoice details in the order or billing section of your account.",
    "check_payment_methods": "Available payment methods are normally shown during checkout. Please check the payment options offered for your region and order.",
    "check_refund_policy": "Refund eligibility and timing depend on the product and payment method. Please review the refund policy associated with your order.",
    "complaint": "I'm sorry you're having this experience. Please describe the issue and include your order or case number if you have one, without sharing passwords or full payment details.",
    "contact_customer_service": "You can reach customer service through the Help or Contact Us section of the service's official website or app.",
    "contact_human_agent": "For a human agent, use the official support channel and request a representative. Keep your order or case number handy.",
    "create_account": "To create an account, choose Sign Up or Create Account on the official website or app and follow the instructions.",
    "delete_account": "Account deletion is usually available in account settings or through official customer support. Check any data-retention or subscription implications first.",
    "delivery_options": "Delivery options and charges are usually displayed at checkout and depend on your address and the items in your order.",
    "delivery_period": "The estimated delivery window is usually available in your order details or shipping confirmation. Delays can affect the estimate.",
    "edit_account": "Open account settings to edit the profile information that can be changed. Some details may require verification or support assistance.",
    "get_invoice": "Look in your account's order history or billing section for an invoice download option. Availability can depend on the order status.",
    "get_refund": "To request a refund, open the relevant order and look for a return or refund option. The process and eligibility depend on the order's policy.",
    "newsletter_subscription": "You can usually subscribe or unsubscribe using the newsletter option in your account or the link at the bottom of a newsletter email.",
    "payment_issue": "Please verify your payment details and check with your bank or payment provider for declined transactions. Avoid sharing full card details here.",
    "place_order": "To place an order, add the items to your cart, continue to checkout, confirm delivery details, and select an available payment method.",
    "recover_password": "Use the Forgot Password option on the sign-in page and follow the password-reset instructions sent to your registered email or phone.",
    "registration_problems": "If registration fails, check that required fields are complete and your email or phone is valid. Try again or contact official support if it continues.",
    "review": "You can usually leave a review from the product page or your order history after purchase. Follow the site's review guidelines.",
    "set_up_shipping_address": "Open your account's address book or checkout page and choose Add Address. Double-check the postcode and contact information before saving.",
    "switch_account": "Use the profile or account menu to sign out, then sign in with the other account. Make sure you are using the correct login details.",
    "track_order": "You can track your order from the order history or shipping confirmation using the tracking link or tracking number.",
    "track_refund": "Check the order's refund status in your account. Processing time depends on the seller, bank, and original payment method.",
}


@lru_cache(maxsize=1)
def _load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Model files are missing. Run: python src/train_model.py")
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def get_reply(message: str) -> tuple[str, str, float]:
    """Return a reply, predicted intent, and model confidence."""
    clean = clean_text(message)
    if not clean:
        return "Please type a question so I can help.", "empty_message", 0.0

    model, vectorizer = _load_artifacts()
    features = vectorizer.transform([clean])
    probabilities = model.predict_proba(features)[0]
    best_idx = probabilities.argmax()
    intent = str(model.classes_[best_idx])
    confidence = float(probabilities[best_idx])

    if confidence < 0.22:
        reply = "I'm not fully sure I understood. Could you rephrase your question or contact customer support for help?"
    else:
        reply = RESPONSES.get(intent, "I can help with common customer service questions. Could you share a little more detail?")
    return reply, intent, confidence
