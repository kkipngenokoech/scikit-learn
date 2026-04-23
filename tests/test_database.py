import pytest
from datetime import datetime
from src.core.database import Database
from src.core.models import User, Post, Comment


@pytest.fixture
def db():
    return Database()


@pytest.fixture
def db_with_user(db):
    user = db.create_user("testuser", "test@example.com")
    return db, user


@pytest.fixture
def db_with_post(db_with_user):
    db, user = db_with_user
    post = db.create_post("Test Post", "Test content", user.id)
    return db, user, post


class TestUserOperations:
    def test_create_user(self, db):
        user = db.create_user("testuser", "test@example.com")
        assert user.id == 1
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True
        assert isinstance(user.created_at, datetime)

    def test_get_user(self, db_with_user):
        db, user = db_with_user
        retrieved_user = db.get_user(user.id)
        assert retrieved_user == user

    def test_get_nonexistent_user(self, db):
        user = db.get_user(999)
        assert user is None

    def test_get_user_by_username(self, db_with_user):
        db, user = db_with_user
        retrieved_user = db.get_user_by_username("testuser")
        assert retrieved_user == user

    def test_get_user_by_nonexistent_username(self, db):
        user = db.get_user_by_username("nonexistent")
        assert user is None

    def test_get_all_users(self, db):
        user1 = db.create_user("user1", "user1@example.com")
        user2 = db.create_user("user2", "user2@example.com")
        users = db.get_all_users()
        assert len(users) == 2
        assert user1 in users
        assert user2 in users


class TestPostOperations:
    def test_create_post(self, db_with_user):
        db, user = db_with_user
        post = db.create_post("Test Post", "Test content", user.id)
        assert post.id == 1
        assert post.title == "Test Post"
        assert post.content == "Test content"
        assert post.author_id == user.id
        assert isinstance(post.created_at, datetime)

    def test_create_post_invalid_author(self, db):
        post = db.create_post("Test Post", "Test content", 999)
        assert post is None

    def test_get_post(self, db_with_post):
        db, user, post = db_with_post
        retrieved_post = db.get_post(post.id)
        assert retrieved_post == post

    def test_get_nonexistent_post(self, db):
        post = db.get_post(999)
        assert post is None

    def test_get_posts_by_author(self, db_with_user):
        db, user = db_with_user
        post1 = db.create_post("Post 1", "Content 1", user.id)
        post2 = db.create_post("Post 2", "Content 2", user.id)
        posts = db.get_posts_by_author(user.id)
        assert len(posts) == 2
        assert post1 in posts
        assert post2 in posts

    def test_get_all_posts(self, db_with_user):
        db, user = db_with_user
        post1 = db.create_post("Post 1", "Content 1", user.id)
        post2 = db.create_post("Post 2", "Content 2", user.id)
        posts = db.get_all_posts()
        assert len(posts) == 2
        assert post1 in posts
        assert post2 in posts

    def test_update_post(self, db_with_post):
        db, user, post = db_with_post
        success = db.update_post(post.id, title="Updated Title", content="Updated content")
        assert success is True
        updated_post = db.get_post(post.id)
        assert updated_post.title == "Updated Title"
        assert updated_post.content == "Updated content"
        assert updated_post.updated_at is not None

    def test_update_nonexistent_post(self, db):
        success = db.update_post(999, title="Updated Title")
        assert success is False

    def test_delete_post(self, db_with_post):
        db, user, post = db_with_post
        success = db.delete_post(post.id)
        assert success is True
        assert db.get_post(post.id) is None

    def test_delete_nonexistent_post(self, db):
        success = db.delete_post(999)
        assert success is False


class TestCommentOperations:
    def test_create_comment(self, db_with_post):
        db, user, post = db_with_post
        comment = db.create_comment("Test comment", post.id, user.id)
        assert comment.id == 1
        assert comment.content == "Test comment"
        assert comment.post_id == post.id
        assert comment.author_id == user.id
        assert comment.parent_id is None
        assert isinstance(comment.created_at, datetime)

    def test_create_comment_invalid_post(self, db_with_user):
        db, user = db_with_user
        comment = db.create_comment("Test comment", 999, user.id)
        assert comment is None

    def test_create_comment_invalid_author(self, db_with_post):
        db, user, post = db_with_post
        comment = db.create_comment("Test comment", post.id, 999)
        assert comment is None

    def test_create_reply_comment(self, db_with_post):
        db, user, post = db_with_post
        parent_comment = db.create_comment("Parent comment", post.id, user.id)
        reply = db.create_comment("Reply comment", post.id, user.id, parent_comment.id)
        assert reply.parent_id == parent_comment.id

    def test_create_reply_invalid_parent(self, db_with_post):
        db, user, post = db_with_post
        reply = db.create_comment("Reply comment", post.id, user.id, 999)
        assert reply is None

    def test_get_comment(self, db_with_post):
        db, user, post = db_with_post
        comment = db.create_comment("Test comment", post.id, user.id)
        retrieved_comment = db.get_comment(comment.id)
        assert retrieved_comment == comment

    def test_get_nonexistent_comment(self, db):
        comment = db.get_comment(999)
        assert comment is None

    def test_get_comments_by_post(self, db_with_post):
        db, user, post = db_with_post
        comment1 = db.create_comment("Comment 1", post.id, user.id)
        comment2 = db.create_comment("Comment 2", post.id, user.id)
        comments = db.get_comments_by_post(post.id)
        assert len(comments) == 2
        assert comment1 in comments
        assert comment2 in comments

    def test_delete_comment(self, db_with_post):
        db, user, post = db_with_post
        comment = db.create_comment("Test comment", post.id, user.id)
        success = db.delete_comment(comment.id)
        assert success is True
        assert db.get_comment(comment.id) is None

    def test_delete_nonexistent_comment(self, db):
        success = db.delete_comment(999)
        assert success is False

    def test_delete_post_cascades_comments(self, db_with_post):
        db, user, post = db_with_post
        comment = db.create_comment("Test comment", post.id, user.id)
        db.delete_post(post.id)
        assert db.get_comment(comment.id) is None
