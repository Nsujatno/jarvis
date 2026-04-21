from pydantic import BaseModel, Field
from typing import Optional

class NotionPage(BaseModel):
    name: str
    text: str
    course_class: str
    due_date: str
    status: str
    status_color: str
    urgency: str
    notion_url: str
