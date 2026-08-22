document.addEventListener("DOMContentLoaded", function () {

const form = document.getElementById("messageForm");
const input = document.getElementById("userInput");
const button = document.getElementById("sendButton");
const chatBox = document.getElementById("chatBox");
const newChatButton = document.getElementById("newChatButton");
const conversationList = document.getElementById("conversationList");
const mobileMenuButton = document.getElementById("mobileMenuButton");
const sidebar = document.getElementById("sidebar");

let currentConversationId = null;
let isGenerating = false;


/* =====================================================
SCROLL
===================================================== */

function scrollBottom() {

chatBox.scrollTop = chatBox.scrollHeight;

}


/* =====================================================
ADD USER MESSAGE
===================================================== */

function addUserMessage(text) {

const messageDiv = document.createElement("div");

messageDiv.className = "chat-message user";

const bubble = document.createElement("div");

bubble.className = "bubble";

bubble.textContent = text;

messageDiv.appendChild(bubble);

chatBox.appendChild(messageDiv);

scrollBottom();

}


/* =====================================================
CREATE BOT MESSAGE
===================================================== */

function createBotMessage() {

const messageDiv = document.createElement("div");

messageDiv.className = "chat-message bot";

const icon = document.createElement("img");

icon.src =
"https://cdn-icons-png.flaticon.com/512/387/387569.png";

icon.className = "bot-icon";

icon.alt = "Medical Assistant";

const bubble = document.createElement("div");

bubble.className = "bubble";

bubble.textContent = "";

messageDiv.appendChild(icon);

messageDiv.appendChild(bubble);

chatBox.appendChild(messageDiv);

scrollBottom();

return bubble;

}


/* =====================================================
CLEAR CHAT
===================================================== */

function clearChat() {

chatBox.innerHTML = "";

}


/* =====================================================
WELCOME MESSAGE
===================================================== */

function showWelcome() {

const messageDiv = document.createElement("div");

messageDiv.className = "chat-message bot";

const icon = document.createElement("img");

icon.src =
"https://cdn-icons-png.flaticon.com/512/387/387569.png";

icon.className = "bot-icon";

icon.alt = "Bot";

const bubble = document.createElement("div");

bubble.className = "bubble";

bubble.textContent =
"Hello! Ask me a medical question.";

messageDiv.appendChild(icon);

messageDiv.appendChild(bubble);

chatBox.appendChild(messageDiv);

}


/* =====================================================
LOAD CONVERSATION
===================================================== */

async function loadConversation(id) {

try {

const response = await fetch(
"/conversation/" + id
);

const data = await response.json();

if (!data.success) {
return;
}

currentConversationId = id;

clearChat();

data.messages.forEach(function (message) {

if (message.role === "user") {

addUserMessage(
message.content
);

}
else if (message.role === "assistant") {

const bubble =
createBotMessage();

bubble.textContent =
message.content;

}

});

document
.querySelectorAll(".conversation-item")
.forEach(function (item) {

item.classList.remove("active");

});

const selected =
document.querySelector(
`.conversation-item[data-id="${id}"]`
);

if (selected) {

selected.classList.add("active");

}

if (window.innerWidth <= 800) {

sidebar.classList.remove("open");

}

}
catch (error) {

console.error(
"Conversation loading error:",
error
);

}

}


/* =====================================================
ADD CONVERSATION TO SIDEBAR
===================================================== */

function addConversationToSidebar(
id,
title
) {

const existing =
document.querySelector(
`.conversation-item[data-id="${id}"]`
);

if (existing) {

const name =
existing.querySelector(
".conversation-name"
);

if (name) {
name.textContent = title;
}

return;

}

const item =
document.createElement("div");

item.className =
"conversation-item";

item.dataset.id = id;

const info =
document.createElement("div");

info.className =
"conversation-info";

const name =
document.createElement("span");

name.className =
"conversation-name";

name.textContent = title;

info.appendChild(name);

const deleteButton =
document.createElement("button");

deleteButton.className =
"delete-btn";

deleteButton.dataset.id = id;

deleteButton.title =
"Delete conversation";

deleteButton.textContent = "×";

item.appendChild(info);

item.appendChild(deleteButton);

conversationList.prepend(item);

}


/* =====================================================
NEW CHAT
===================================================== */

async function createNewChat() {

if (isGenerating) {
return;
}

try {

const response =
await fetch(
"/new_chat",
{
method: "POST"
}
);

const data =
await response.json();

if (!data.success) {
return;
}

currentConversationId =
data.conversation_id;

clearChat();

showWelcome();

addConversationToSidebar(
data.conversation_id,
data.title
);

document
.querySelectorAll(".conversation-item")
.forEach(function (item) {

item.classList.remove("active");

});

const newItem =
document.querySelector(
`.conversation-item[data-id="${data.conversation_id}"]`
);

if (newItem) {
newItem.classList.add("active");
}

input.focus();

}
catch (error) {

console.error(
"New chat error:",
error
);

}

}


/* =====================================================
SEND MESSAGE
===================================================== */

async function sendMessage() {

if (isGenerating) {
return;
}

const message =
input.value.trim();

if (!message) {
return;
}

isGenerating = true;

button.disabled = true;

input.disabled = true;


/* -----------------------------------------------------
CREATE THREAD IF NEEDED
----------------------------------------------------- */

if (!currentConversationId) {

try {

const response =
await fetch(
"/new_chat",
{
method: "POST"
}
);

const data =
await response.json();

currentConversationId =
data.conversation_id;

addConversationToSidebar(
currentConversationId,
"New Chat"
);

}
catch (error) {

console.error(
"Thread creation error:",
error
);

isGenerating = false;

button.disabled = false;

input.disabled = false;

return;

}

}


/* -----------------------------------------------------
DISPLAY USER MESSAGE
----------------------------------------------------- */

addUserMessage(message);

input.value = "";


/* -----------------------------------------------------
CREATE BOT BUBBLE
----------------------------------------------------- */

const botBubble =
createBotMessage();


try {

const formData =
new FormData();

formData.append(
"msg",
message
);

formData.append(
"conversation_id",
currentConversationId
);


/* -----------------------------------------------------
SEND TO FLASK
----------------------------------------------------- */

const response =
await fetch(
"/chat",
{
method: "POST",
body: formData
}
);

if (!response.ok) {

throw new Error(
"Server error: " +
response.status
);

}

if (!response.body) {

throw new Error(
"Streaming response not available."
);

}


/* -----------------------------------------------------
STREAM RESPONSE
----------------------------------------------------- */

const reader =
response.body.getReader();

const decoder =
new TextDecoder("utf-8");

let fullText = "";

while (true) {

const result =
await reader.read();

if (result.done) {
break;
}

const chunk =
decoder.decode(
result.value,
{
stream: true
}
);

if (chunk) {

fullText += chunk;

botBubble.textContent =
fullText;

scrollBottom();

}

}


/* -----------------------------------------------------
FINAL DECODER
----------------------------------------------------- */

const finalChunk =
decoder.decode();

if (finalChunk) {

fullText += finalChunk;

botBubble.textContent =
fullText;

}


/* -----------------------------------------------------
UPDATE SIDEBAR TITLE
-----------------------------------------------------

The first question becomes the title.
*/

const conversationItem =
document.querySelector(
`.conversation-item[data-id="${currentConversationId}"]`
);

if (conversationItem) {

const title =
conversationItem.querySelector(
".conversation-name"
);

if (title &&
    title.textContent === "New Chat") {

let shortTitle =
message.replace(/\s+/g, " ").trim();

if (shortTitle.length > 45) {

shortTitle =
shortTitle.substring(0, 45)
+ "...";

}

title.textContent =
shortTitle;

}

}


/* -----------------------------------------------------
MOVE CURRENT CHAT TO TOP
----------------------------------------------------- */

const currentItem =
document.querySelector(
`.conversation-item[data-id="${currentConversationId}"]`
);

if (currentItem) {

conversationList.prepend(
currentItem
);

}

}
catch (error) {

console.error(
"Chat error:",
error
);

botBubble.textContent =
"Sorry, something went wrong. Please try again.";

}
finally {

isGenerating = false;

button.disabled = false;

input.disabled = false;

input.focus();

}

}


/* =====================================================
FORM SUBMIT
===================================================== */

form.addEventListener(
"submit",
function (event) {

event.preventDefault();

sendMessage();

}
);


/* =====================================================
ENTER KEY
===================================================== */

input.addEventListener(
"keydown",
function (event) {

if (
event.key === "Enter" &&
!event.shiftKey
) {

event.preventDefault();

sendMessage();

}

}
);


/* =====================================================
NEW CHAT BUTTON
===================================================== */

newChatButton.addEventListener(
"click",
function () {

createNewChat();

}
);


/* =====================================================
CONVERSATION CLICK
===================================================== */

conversationList.addEventListener(
"click",
function (event) {

const deleteButton =
event.target.closest(
".delete-btn"
);

if (deleteButton) {

event.stopPropagation();

deleteConversation(
deleteButton.dataset.id
);

return;

}

const item =
event.target.closest(
".conversation-item"
);

if (item) {

loadConversation(
item.dataset.id
);

}

}
);


/* =====================================================
DELETE CONVERSATION
===================================================== */

async function deleteConversation(id) {

if (
!confirm(
"Delete this conversation?"
)
) {

return;

}

try {

const response =
await fetch(
"/conversation/" + id,
{
method: "DELETE"
}
);

const data =
await response.json();

if (!data.success) {
return;
}

const item =
document.querySelector(
`.conversation-item[data-id="${id}"]`
);

if (item) {
item.remove();
}


/* -----------------------------------------------------
If current conversation deleted
----------------------------------------------------- */

if (
currentConversationId === id
) {

currentConversationId =
null;

clearChat();

showWelcome();

}

}
catch (error) {

console.error(
"Delete error:",
error
);

}

}


/* =====================================================
MOBILE SIDEBAR
===================================================== */

mobileMenuButton.addEventListener(
"click",
function () {

sidebar.classList.toggle(
"open"
);

}
);


/* =====================================================
LOAD FIRST CONVERSATION
===================================================== */

const firstConversation =
document.querySelector(
".conversation-item"
);

if (firstConversation) {

loadConversation(
firstConversation.dataset.id
);

}
else {

showWelcome();

}

});