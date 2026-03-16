import { useState, useEffect, useCallback } from "react"
import { Sidebar } from "./components/Sidebar"
import { MainContent } from "./components/MainContent"
import { SettingsDialog } from "./components/SettingsDialog"
import { api } from "./lib/api"
import type { Document, Category } from "./types"

interface ChatSession {
  id: number
  title: string
  messages: { role: "user" | "assistant"; content: string }[]
  updatedAt: string
}

function App() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [categories, setCategories] = useState<Category[]>([])
  const [chatHistory, setChatHistory] = useState<ChatSession[]>([])
  const [activeChatId, setActiveChatId] = useState<number | null>(null)
  const [currentMessages, setCurrentMessages] = useState<{ role: "user" | "assistant"; content: string }[]>([])
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [settingsDefaultTab, setSettingsDefaultTab] = useState<"documents" | "models" | "system">("documents")

  const loadDocuments = useCallback(async () => {
    const response = await api.documents.list(0, 100)
    if (response.data) {
      setDocuments(response.data.documents)
    }
  }, [])

  const loadCategories = useCallback(async () => {
    const response = await api.categories.list()
    if (response.data) {
      setCategories(response.data)
    }
  }, [])

  useEffect(() => {
    loadDocuments()
    loadCategories()
    loadChatHistory()
  }, [loadDocuments, loadCategories])

  const loadChatHistory = () => {
    const saved = localStorage.getItem("chatHistory")
    if (saved) {
      try {
        setChatHistory(JSON.parse(saved))
      } catch {
        setChatHistory([])
      }
    }
  }

  const saveChatHistory = (history: ChatSession[]) => {
    localStorage.setItem("chatHistory", JSON.stringify(history))
    setChatHistory(history)
  }

  const handleNewChat = () => {
    setActiveChatId(null)
    setCurrentMessages([])
  }

  const handleChatSelect = (id: number) => {
    const chat = chatHistory.find(c => c.id === id)
    if (chat) {
      setActiveChatId(chat.id)
      setCurrentMessages(chat.messages)
    }
  }

  const handleChatDelete = (id: number) => {
    const newHistory = chatHistory.filter(c => c.id !== id)
    saveChatHistory(newHistory)
    if (activeChatId === id) {
      handleNewChat()
    }
  }

  const handleSendMessage = async (content: string) => {
    const userMsg = { role: "user" as const, content }
    setCurrentMessages(prev => [...prev, userMsg])

    const response = await api.qa.ask(content)
    
    if (response.data) {
      const assistantMsg = { role: "assistant" as const, content: response.data.answer }
      setCurrentMessages(prev => [...prev, assistantMsg])

      const chatId = activeChatId || Date.now()
      const newChat: ChatSession = {
        id: chatId,
        title: content.slice(0, 30) + (content.length > 30 ? "..." : ""),
        messages: [...currentMessages, userMsg, { role: "assistant", content: response.data.answer }],
        updatedAt: new Date().toISOString()
      }

      if (activeChatId) {
        const updated = chatHistory.map(c => c.id === activeChatId ? newChat : c)
        saveChatHistory(updated)
      } else {
        saveChatHistory([newChat, ...chatHistory])
        setActiveChatId(chatId)
      }
    }
  }

  const handleOpenDocumentSettings = () => {
    setSettingsDefaultTab("documents")
    setSettingsOpen(true)
  }

  return (
    <div className="flex h-screen bg-background">
      <Sidebar
        onNewChat={handleNewChat}
        onSettingsClick={() => setSettingsOpen(true)}
        chatHistory={chatHistory}
        onChatSelect={handleChatSelect}
        onChatDelete={handleChatDelete}
        activeChatId={activeChatId}
      />
      <MainContent
        messages={currentMessages}
        onSendMessage={handleSendMessage}
        onOpenDocumentSettings={handleOpenDocumentSettings}
      />
      <SettingsDialog
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        documents={documents}
        categories={categories}
        onDocumentsChange={loadDocuments}
        onCategoriesChange={loadCategories}
        defaultTab={settingsDefaultTab}
      />
    </div>
  )
}

export default App
