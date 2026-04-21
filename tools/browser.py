import webbrowser

def open_url(url: str) -> str:
    """
    Opens a specified URL in the user's default web browser.
    
    Args:
        url (str): The complete URL to open.
    Returns:
        str: A message confirming the action was triggered.
    """
    webbrowser.open(url)
    return f"Opening {url} in your browser..."
