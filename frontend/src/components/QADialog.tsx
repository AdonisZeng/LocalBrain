import { useState, useRef, useEffect } from "react"
import { Send, Bot, FileText, Loader2, Sparkles, User } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Dialog, DialogContent } from "@/components/ui/dialog"
import { api } from "@/lib/api"
import type { QASource } from "@/types"

interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  sources?: QASource[]
}

interface QADialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function QADialog({ open, onOpenChange }: QADialogProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (open && messages.length === 0) {
      setMessages([
        {
          id: "welcome",
          role: "assistant",
          content: "你好！我是 LocalBrain 智能助手。你可以向我提问，我会根据知识库中的文档来回答你的问题。",
        },
      ])
    }
  }, [open, messages.length])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleSend = async () => {
    if (!input.trim() || loading) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: input,
    }
    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setLoading(true)

    const response = await api.qa.ask(input)

    setLoading(false)

    if (response.data) {
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.data.answer,
        sources: response.data.sources,
      }
      setMessages((prev) => [...prev, assistantMessage])
    } else if (response.error) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `抱歉，发生错误：${response.error}`,
      }
      setMessages((prev) => [...prev, errorMessage])
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl h-[650px] p-0 flex flex-col animate-scale-in">
        <div className="p-4 bg-gradient-to-r from-primary/5 to-transparent relative">
          <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-primary/20 via-border/30 to-transparent" />
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-primary/70 flex items-center justify-center shadow-lg shadow-primary/20">
              <Sparkles className="h-5 w-5 text-primary-foreground" />
            </div>
            <div>
              <h2 className="text-lg font-semibold">智能问答</h2>
              <p className="text-xs text-muted-foreground">基于本地知识库的 AI 助手</p>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-auto p-4 space-y-4 scrollbar-thin">
          {messages.map((message, idx) => (
            <div
              key={message.id}
              className={`flex gap-3 animate-fade-in-up ${message.role === "user" ? "flex-row-reverse" : ""}`}
              style={{ animationDelay: `${Math.min(idx * 0.05, 0.3)}s` }}
            >
              <div
                className={`w-9 h-9 rounded-full flex items-center justify-center shrink-0 transition-transform duration-300 hover:scale-110 ${
                  message.role === "user" 
                    ? "bg-gradient-to-br from-primary to-primary/80 text-primary-foreground shadow-lg shadow-primary/20" 
                    : "bg-muted border border-border"
                }`}
              >
                {message.role === "user" 
                  ? <User className="h-4 w-4" /> 
                  : <Bot className="h-4 w-4 text-primary" />}
              </div>
              <div
                className={`group max-w-[80%] ${message.role === "user" ? "message-bubble" : ""}`}
              >
                <div
                  className={`rounded-2xl px-4 py-3 transition-all duration-200 ${
                    message.role === "user"
                      ? "bg-gradient-to-br from-primary to-primary/90 text-primary-foreground shadow-lg shadow-primary/10"
                      : "bg-muted border border-border/50 hover:border-border"
                  }`}
                >
                  <div className="whitespace-pre-wrap text-[15px] leading-relaxed">{message.content}</div>
                  {message.sources && message.sources.length > 0 && (
                    <div className="mt-3 pt-3 relative">
                      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-border/20 to-transparent" />
                      <div className="text-xs font-semibold mb-2 opacity-70">参考来源:</div>
                      <div className="space-y-1.5">
                        {message.sources.map((source, sidx) => (
                          <div 
                            key={sidx} 
                            className="flex items-center gap-2 text-xs opacity-80 hover:opacity-100 transition-opacity duration-200 cursor-default"
                          >
                            <div className="w-5 h-5 rounded bg-primary/10 flex items-center justify-center shrink-0">
                              <FileText className="h-3 w-3 text-primary" />
                            </div>
                            <span className="truncate">{source.title}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                {message.role === "assistant" && (
                  <div className="absolute -bottom-5 left-0 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                    <span className="text-[10px] text-muted-foreground">AI 生成内容</span>
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex gap-3 animate-fade-in">
              <div className="w-9 h-9 rounded-full bg-muted border border-border flex items-center justify-center">
                <Bot className="h-4 w-4 text-primary" />
              </div>
              <div className="bg-muted border border-border/50 rounded-2xl px-4 py-3">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} className="h-4" />
        </div>

        <div className="p-4 bg-card/80 backdrop-blur-md relative">
          <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-border/50 to-transparent" />
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <Input
                placeholder="输入你的问题..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={loading}
                className="h-11 pr-12 transition-all duration-300 focus:ring-2 focus:ring-primary/30"
              />
              <div className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-muted-foreground pointer-events-none">
                {input.length > 0 && <span className="opacity-60">{input.length}</span>}
              </div>
            </div>
            <Button 
              onClick={handleSend} 
              disabled={loading || !input.trim()}
              className="h-11 px-5 transition-all duration-300 hover:shadow-lg hover:shadow-primary/20 hover:scale-[1.02] active:scale-[0.98] disabled:hover:scale-100"
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4 transition-transform duration-200 group-hover:translate-x-0.5" />
              )}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground text-center mt-2 opacity-70">
            按 <kbd className="px-1.5 py-0.5 bg-muted rounded text-[10px] font-mono">Enter</kbd> 发送
          </p>
        </div>
      </DialogContent>
    </Dialog>
  )
}
