import { useState, useRef, useEffect } from "react"
import { Upload, Send, Bot, User, Loader2, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

interface ChatMessage {
  role: "user" | "assistant"
  content: string
}

interface MainContentProps {
  messages: ChatMessage[]
  onSendMessage: (content: string) => void
  onOpenDocumentSettings?: () => void
}

export function MainContent({ messages, onSendMessage, onOpenDocumentSettings }: MainContentProps) {
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  useEffect(() => {
    if (!loading && inputRef.current) {
      inputRef.current.focus()
    }
  }, [loading])

  const handleSend = async () => {
    if (!input.trim() || loading) return
    
    const userInput = input
    setInput("")
    setLoading(true)
    setIsTyping(true)
    
    await onSendMessage(userInput)
    
    setLoading(false)
    setIsTyping(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const showWelcome = messages.length === 0

  return (
    <div className="flex-1 flex flex-col bg-background">
      <div className="flex-1 overflow-auto p-6 scrollbar-thin">
        <div className="max-w-3xl mx-auto">
          {showWelcome ? (
            <div className="text-center py-16 animate-fade-in-up">
              <div className="relative inline-flex items-center justify-center w-24 h-24 mb-8">
                <div className="absolute inset-0 bg-primary/20 rounded-full animate-pulse" />
                <div className="absolute inset-2 bg-primary/10 rounded-full" />
                <div className="relative w-20 h-20 rounded-full bg-gradient-to-br from-primary to-primary/70 flex items-center justify-center shadow-lg shadow-primary/30">
                  <Sparkles className="h-10 w-10 text-primary-foreground" />
                </div>
              </div>
              <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
                LocalBrain 智能助手
              </h1>
              <p className="text-muted-foreground text-lg mb-10 max-w-md mx-auto">
                基于本地知识库的智能问答系统，你的私人 AI 助理
              </p>
              <div className="text-left max-w-sm mx-auto space-y-4">
                {[
                  { icon: Upload, text: "上传你的知识文档", delay: "stagger-1" },
                  { icon: Send, text: "在下方输入框提问", delay: "stagger-2" },
                  { icon: Bot, text: "获取基于文档的智能回答", delay: "stagger-3" },
                ].map((item, idx) => (
                  <div 
                    key={idx}
                    className={`flex items-center gap-4 p-4 rounded-xl bg-card border border-border/50 hover:border-primary/30 hover:bg-accent/30 transition-all duration-300 cursor-default animate-fade-in-up ${item.delay}`}
                  >
                    <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                      <item.icon className="h-5 w-5 text-primary" />
                    </div>
                    <span className="text-sm text-muted-foreground">{idx + 1}. {item.text}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex gap-4 animate-fade-in-up ${msg.role === "user" ? "flex-row-reverse" : ""}`}
                  style={{ animationDelay: `${Math.min(idx * 0.05, 0.3)}s` }}
                >
                  <div className={`shrink-0 w-10 h-10 rounded-full flex items-center justify-center transition-transform duration-300 hover:scale-110 ${
                    msg.role === "user" 
                      ? "bg-gradient-to-br from-primary to-primary/80 text-primary-foreground shadow-lg shadow-primary/20" 
                      : "bg-muted border border-border"
                  }`}>
                    {msg.role === "user" 
                      ? <User className="h-5 w-5" /> 
                      : <Bot className="h-5 w-5 text-primary" />}
                  </div>
                  <div className={`group relative max-w-[75%] ${
                    msg.role === "user" ? "message-bubble" : ""
                  }`}>
                    <div className={`rounded-2xl px-5 py-4 transition-all duration-200 ${
                      msg.role === "user"
                        ? "bg-gradient-to-br from-primary to-primary/90 text-primary-foreground shadow-lg shadow-primary/10"
                        : "bg-muted border border-border/50 hover:border-border"
                    }`}>
                      <div className="whitespace-pre-wrap text-[15px] leading-relaxed">{msg.content}</div>
                    </div>
                    {msg.role === "assistant" && (
                      <div className="absolute -bottom-6 left-0 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                        <span className="text-xs text-muted-foreground">AI 生成内容</span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="flex gap-4 animate-fade-in">
                  <div className="shrink-0 w-10 h-10 rounded-full bg-muted border border-border flex items-center justify-center">
                    <Bot className="h-5 w-5 text-primary" />
                  </div>
                  <div className="bg-muted border border-border/50 rounded-2xl px-5 py-4">
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
          )}
        </div>
      </div>

      <div className="bg-card/80 backdrop-blur-md sticky bottom-0">
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-border/60 to-transparent" />
        <div className="max-w-3xl mx-auto p-4">
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <Input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="输入你的问题..."
                className={`w-full h-12 text-base pr-12 transition-all duration-300 focus:ring-2 focus:ring-primary/30 ${
                  isTyping ? "animate-pulse" : ""
                }`}
                disabled={loading}
              />
              <div className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-muted-foreground pointer-events-none">
                {input.length > 0 && <span className="opacity-60">{input.length}</span>}
              </div>
            </div>
            <Button 
              onClick={handleSend} 
              size="lg" 
              disabled={loading || !input.trim()}
              className="h-12 px-6 transition-all duration-300 hover:shadow-lg hover:shadow-primary/20 hover:scale-[1.02] active:scale-[0.98] disabled:hover:scale-100"
            >
              {loading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Send className="h-5 w-5 transition-transform duration-200 group-hover:translate-x-0.5" />
              )}
            </Button>
            <Button 
              variant="outline" 
              size="lg" 
              onClick={() => onOpenDocumentSettings?.()}
              className="h-12 px-4 transition-all duration-300 hover:bg-primary/5 hover:border-primary/30 hover:text-primary"
            >
              <Upload className="h-5 w-5" />
            </Button>
          </div>
          <p className="text-xs text-muted-foreground text-center mt-3 opacity-70">
            按 <kbd className="px-1.5 py-0.5 bg-muted rounded text-[10px] font-mono">Enter</kbd> 发送 · 
            <kbd className="px-1.5 py-0.5 bg-muted rounded text-[10px] font-mono">Shift+Enter</kbd> 换行
          </p>
        </div>
      </div>
    </div>
  )
}
