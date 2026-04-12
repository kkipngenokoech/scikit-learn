from typing import List, Optional, Dict, Any
from datetime import datetime
from .models import User, Post, Comment


class Database:
    def __init__(self):
        self._users: Dict[int, User] = {}
        self._posts: Dict[int, Post] = {}
        self._comments: Dict[int, Comment] = {}
        self._next_user_id = 1
        self._next_post_id = 1
        self._next_comment_id = 1

    def create_user(self, username: str, email: str) -> User:
        user = User(
            id=self._next_user_id,
            username=username,
            email=email,
            created_at=datetime.now()
        )
        self._users[user.id] = user
        self._next_user_id += 1
        return user

    def get_user(self, user_id: int) -> Optional[User]:
        return self._users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        for user in self._users.values():
            if user.username == username:
                return user
        return None

    def get_all_users(self) -> List[User]:
        return list(self._users.values())

    def create_post(self, title: str, content: str, author_id: int) -> Optional[Post]:
        if author_id not in self._users:
            return None
        
        post = Post(
            id=self._next_post_id,
            title=title,
            content=content,
            author_id=author_id,
            created_at=datetime.now()
        )
        self._posts[post.id] = post
        self._next_post_id += 1
        return post

    def get_post(self, post_id: int) -> Optional[Post]:
        return self._posts.get(post_id)

    def get_posts_by_author(self, author_id: int) -> List[Post]:
        return [post for post in self._posts.values() if post.author_id == author_id]

    def get_all_posts(self) -> List[Post]:
        return list(self._posts.values())

    def update_post(self, post_id: int, title: Optional[str] = None, content: Optional[str] = None) -> bool:
        post = self._posts.get(post_id)
        if not post:
            return False
        
        if title is not None:
            post.title = title
        if content is not None:
            post.content = content
        post.updated_at = datetime.now()
        return True

    def delete_post(self, post_id: int) -> bool:
        if post_id in self._posts:
            del self._posts[post_id]
            # Also delete associated comments
            comments_to_delete = [cid for cid, comment in self._comments.items() if comment.post_id == post_id]
            for cid in comments_to_delete:
                del self._comments[cid]
            return True
        return False

    def create_comment(self, content: str, post_id: int, author_id: int, parent_id: Optional[int] = None) -> Optional[Comment]:
        if post_id not in self._posts or author_id not in self._users:
            return None
        
        if parent_id is not None and parent_id not in self._comments:
            return None
        
        comment = Comment(
            id=self._next_comment_id,
            content=content,
            post_id=post_id,
            author_id=author_id,
            created_at=datetime.now(),
            parent_id=parent_id
        )
        self._comments[comment.id] = comment
        self._next_comment_id += 1
        return comment

    def get_comment(self, comment_id: int) -> Optional[Comment]:
        return self._comments.get(comment_id)

    def get_comments_by_post(self, post_id: int) -> List[Comment]:
        return [comment for comment in self._comments.values() if comment.post_id == post_id]

    def delete_comment(self, comment_id: int) -> bool:
        if comment_id in self._comments:
            del self._comments[comment_id]
            return True
        return False
