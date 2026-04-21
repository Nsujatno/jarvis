from models.notion import NotionPage

def extract_properties(page, text) -> NotionPage:
    if not page:
        return None

    props = page.get("properties", {}) or {}
    
    # get title
    title_list = (props.get("Name") or {}).get("title") or []
    name = title_list[0].get("plain_text", "Untitled") if title_list else "Untitled"
    
    # get class
    class_obj = (props.get("class") or {}).get("select") or {}
    class_name = class_obj.get("name", "Unknown")
    
    # get due date
    date_obj = (props.get("Due date") or {}).get("date") or {}
    due_date = date_obj.get("start", "No Date")
    
    # get status and color
    status_root = (props.get("Status") or {}).get("status") or {}
    status = status_root.get("name", "Unknown")
    status_color = status_root.get("color", "default")
    
    # get days remaining
    formula_obj = (props.get("days remaining") or {}).get("formula") or {}
    urgency = formula_obj.get("string", "N/A")
    
    return NotionPage(
        name=name,
        text=text,
        course_class=class_name,
        due_date=due_date,
        status=status,
        status_color=status_color,
        urgency=urgency,
        notion_url=page.get("url")
    )
