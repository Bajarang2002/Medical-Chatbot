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
       SCROLL TO BOTTOM
    ===================================================== */

    function scrollBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }


    /* =====================================================
       ESCAPE HTML
       Prevents model/user text from injecting HTML.
    ===================================================== */

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }


    /* =====================================================
       FORMAT ASSISTANT RESPONSE
       
       Supports:
       - Bold
       - Headings
       - Bullet points
       - Numbered lists
       - Paragraphs
       - Line breaks
       - Inline code
       
       Does NOT allow arbitrary HTML from the model.
    ===================================================== */

    function formatAssistantResponse(text) {

        if (!text) {
            return "";
        }

        let html = escapeHtml(text);

        /* -------------------------------------------------
           Normalize line endings
        ------------------------------------------------- */

        html = html.replace(/\r\n/g, "\n");
        html = html.replace(/\r/g, "\n");


        /* -------------------------------------------------
           Markdown headings
           # Heading
           ## Heading
           ### Heading
        ------------------------------------------------- */

        html = html.replace(
            /^###\s+(.+)$/gm,
            '<h4>$1</h4>'
        );

        html = html.replace(
            /^##\s+(.+)$/gm,
            '<h3>$1</h3>'
        );

        html = html.replace(
            /^#\s+(.+)$/gm,
            '<h3>$1</h3>'
        );


        /* -------------------------------------------------
           Bold
           **text**
        ------------------------------------------------- */

        html = html.replace(
            /\*\*(.+?)\*\*/g,
            "<strong>$1</strong>"
        );


        /* -------------------------------------------------
           Italic
           *text*
        ------------------------------------------------- */

        html = html.replace(
            /(^|[^\*])\*([^*\n]+)\*(?!\*)/g,
            "$1<em>$2</em>"
        );


        /* -------------------------------------------------
           Inline code
           `text`
        ------------------------------------------------- */

        html = html.replace(
            /`([^`]+)`/g,
            "<code>$1</code>"
        );


        /* -------------------------------------------------
           Bullet lists
           - item
           * item
        ------------------------------------------------- */

        html = html.replace(
            /(?:^|\n)(?:[-*])\s+(.+)(?=\n|$)/g,
            '<li>$1</li>'
        );

        html = html.replace(
            /(<li>.*<\/li>)/gs,
            function (match) {
                return "<ul>" + match + "</ul>";
            }
        );


        /* -------------------------------------------------
           Numbered lists
           1. item
           2. item
        ------------------------------------------------- */

        html = html.replace(
            /(?:^|\n)\d+\.\s+(.+)(?=\n|$)/g,
            '<li>$1</li>'
        );


        /* -------------------------------------------------
           Convert remaining line breaks
        ------------------------------------------------- */

        html = html.replace(
            /\n{2,}/g,
            "</p><p>"
        );

        html = html.replace(
            /\n/g,
            "<br>"
        );


        /*
         * Clean unwanted paragraph wrapping around lists
         */

        html = html.replace(
            /<p>\s*(<ul>)/g,
            "$1"
        );

        html = html.replace(
            /(<\/ul>)\s*<\/p>/g,
            "$1"
        );


        return "<p>" + html + "</p>";
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

        bubble.className = "bubble assistant-response";

        bubble.innerHTML = "";

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

        icon.alt = "Medical Assistant";

        const bubble = document.createElement("div");

        bubble.className = "bubble assistant-response";

        bubble.innerHTML = `
            <p>
                Hello! I'm your <strong>Medical Assistant</strong>.
            </p>

            <p>
                You can ask me questions about the medical information
                available in the provided medical documents.
            </p>

            <p class="response-note">
             Please remember that this assistant provides educational
            information and does not replace professional medical advice.
            </p>
        `;

        messageDiv.appendChild(icon);
        messageDiv.appendChild(bubble);

        chatBox.appendChild(messageDiv);

        scrollBottom();
    }


    /* =====================================================
       LOAD CONVERSATION
    ===================================================== */

    async function loadConversation(id) {

        try {

            const response =
                await fetch("/conversation/" + encodeURIComponent(id));

            if (!response.ok) {
                throw new Error(
                    "Failed to load conversation."
                );
            }

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

                    bubble.innerHTML =
                        formatAssistantResponse(
                            message.content
                        );
                }

            });


            /* -------------------------------------------------
               Active conversation
            ------------------------------------------------- */

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


            /* -------------------------------------------------
               Close mobile sidebar
            ------------------------------------------------- */

            if (window.innerWidth <= 800) {

                sidebar.classList.remove("open");

            }

            scrollBottom();

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

    function addConversationToSidebar(id, title) {

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

        deleteButton.setAttribute(
            "aria-label",
            "Delete conversation"
        );

        deleteButton.textContent = "×";


        item.appendChild(info);

        item.appendChild(deleteButton);


        conversationList.prepend(item);
    }


    /* =====================================================
       CREATE NEW CHAT
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


            if (!response.ok) {

                throw new Error(
                    "Unable to create new chat."
                );

            }


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

        button.classList.add("loading");


        /* -------------------------------------------------
           Create conversation if required
        ------------------------------------------------- */

        if (!currentConversationId) {

            try {

                const response =
                    await fetch(
                        "/new_chat",
                        {
                            method: "POST"
                        }
                    );


                if (!response.ok) {

                    throw new Error(
                        "Unable to create conversation."
                    );

                }


                const data =
                    await response.json();


                if (!data.success) {

                    throw new Error(
                        "Conversation creation failed."
                    );

                }


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

                button.classList.remove("loading");

                return;
            }
        }


        /* -------------------------------------------------
           Display user message
        ------------------------------------------------- */

        addUserMessage(message);

        input.value = "";


        /* -------------------------------------------------
           Create bot response
        ------------------------------------------------- */

        const botBubble =
            createBotMessage();


        /* -------------------------------------------------
           Show temporary typing indicator
        ------------------------------------------------- */

        botBubble.innerHTML = `
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;


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


            /* -------------------------------------------------
               Send request to Flask
            ------------------------------------------------- */

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
                    "Streaming response is not available."
                );

            }


            /* -------------------------------------------------
               Read streaming response
            ------------------------------------------------- */

            const reader =
                response.body.getReader();


            const decoder =
                new TextDecoder("utf-8");


            let fullText = "";


            botBubble.innerHTML = "";


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


                    /*
                     * Render Markdown-like formatting
                     * while streaming.
                     */

                    botBubble.innerHTML =
                        formatAssistantResponse(
                            fullText
                        );


                    scrollBottom();
                }
            }


            /* -------------------------------------------------
               Flush decoder
            ------------------------------------------------- */

            const finalChunk =
                decoder.decode();


            if (finalChunk) {

                fullText += finalChunk;

            }


            botBubble.innerHTML =
                formatAssistantResponse(
                    fullText
                );


            scrollBottom();


            /* -------------------------------------------------
               Update conversation title
            ------------------------------------------------- */

            const conversationItem =
                document.querySelector(
                    `.conversation-item[data-id="${currentConversationId}"]`
                );


            if (conversationItem) {

                const title =
                    conversationItem.querySelector(
                        ".conversation-name"
                    );


                if (
                    title &&
                    title.textContent === "New Chat"
                ) {

                    let shortTitle =
                        message
                            .replace(/\s+/g, " ")
                            .trim();


                    if (shortTitle.length > 45) {

                        shortTitle =
                            shortTitle
                                .substring(0, 45)
                                .trim() + "...";

                    }


                    title.textContent =
                        shortTitle;
                }
            }


            /* -------------------------------------------------
               Move current conversation to top
            ------------------------------------------------- */

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


            botBubble.innerHTML = `
                <div class="error-message">
                    <strong>Unable to generate a response.</strong>
                    <p>
                        Please check your connection and try again.
                    </p>
                </div>
            `;
        }

        finally {

            isGenerating = false;

            button.disabled = false;

            input.disabled = false;

            button.classList.remove("loading");

            input.focus();

            scrollBottom();
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

    if (newChatButton) {

        newChatButton.addEventListener(
            "click",
            function () {

                createNewChat();

            }
        );
    }


    /* =====================================================
       CONVERSATION LIST
    ===================================================== */

    if (conversationList) {

        conversationList.addEventListener(
            "click",
            function (event) {

                /* -------------------------------------------------
                   Delete button
                ------------------------------------------------- */

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


                /* -------------------------------------------------
                   Conversation item
                ------------------------------------------------- */

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
    }


    /* =====================================================
       DELETE CONVERSATION
    ===================================================== */

    async function deleteConversation(id) {

        const confirmed =
            confirm(
                "Are you sure you want to delete this conversation?"
            );


        if (!confirmed) {
            return;
        }


        try {

            const response =
                await fetch(
                    "/conversation/" +
                    encodeURIComponent(id),
                    {
                        method: "DELETE"
                    }
                );


            if (!response.ok) {

                throw new Error(
                    "Delete request failed."
                );

            }


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


            /* -------------------------------------------------
               If active conversation deleted
            ------------------------------------------------- */

            if (
                currentConversationId === id
            ) {

                currentConversationId = null;

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

    if (mobileMenuButton) {

        mobileMenuButton.addEventListener(
            "click",
            function () {

                sidebar.classList.toggle(
                    "open"
                );

            }
        );
    }


    /* =====================================================
       CLOSE SIDEBAR WHEN CLICKING OUTSIDE
       Mobile only
    ===================================================== */

    document.addEventListener(
        "click",
        function (event) {

            if (
                window.innerWidth <= 800 &&
                sidebar.classList.contains("open")
            ) {

                const clickedInsideSidebar =
                    sidebar.contains(event.target);

                const clickedMenuButton =
                    mobileMenuButton &&
                    mobileMenuButton.contains(
                        event.target
                    );


                if (
                    !clickedInsideSidebar &&
                    !clickedMenuButton
                ) {

                    sidebar.classList.remove(
                        "open"
                    );

                }
            }

        }
    );


    /* =====================================================
       INITIALIZE CHAT
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