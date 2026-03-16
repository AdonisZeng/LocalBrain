import { Settings, Plus, MessageSquare, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"

interface ChatSession {
  id: number
  title: string
  updatedAt: string
}

interface SidebarProps {
  onNewChat: () => void
  onSettingsClick: () => void
  chatHistory: ChatSession[]
  onChatSelect: (id: number) => void
  onChatDelete: (id: number) => void
  activeChatId: number | null
}

export function Sidebar({ 
  onNewChat, 
  onSettingsClick, 
  chatHistory, 
  onChatSelect,
  onChatDelete,
  activeChatId 
}: SidebarProps) {
  return (
    <div className="w-64 bg-card flex flex-col h-full animate-slide-in-left relative">
      <div className="absolute right-0 top-0 bottom-0 w-px bg-gradient-to-b from-transparent via-border/50 to-transparent" />
      <div className="p-4 relative">
        <div className="absolute bottom-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-border/30 to-transparent" />
        <Button 
          className="w-full justify-start group relative overflow-hidden transition-all duration-300 hover:shadow-lg hover:scale-[1.02] active:scale-[0.98]" 
          onClick={onNewChat}
        >
          <Plus className="mr-2 h-4 w-4 transition-transform duration-300 group-hover:rotate-90" />
          <span className="relative z-10">新建对话</span>
          <div className="absolute inset-0 bg-gradient-to-r from-primary/0 to-primary/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        </Button>
      </div>

      <div className="flex-1 overflow-auto p-4 scrollbar-thin">
        <h2 className="text-sm font-semibold mb-3 flex items-center text-muted-foreground">
          <MessageSquare className="mr-2 h-4 w-4" />
          对话历史
        </h2>
        {chatHistory.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-muted-foreground animate-fade-in">
            <MessageSquare className="h-12 w-12 mb-3 opacity-30" />
            <p className="text-sm">暂无对话记录</p>
            <p className="text-xs mt-1 opacity-60">开始新对话吧</p>
          </div>
        ) : (
          <ul className="space-y-1">
            {chatHistory.map((chat, index) => (
              <li 
                key={chat.id}
                className={`group flex items-center justify-between rounded-lg transition-all duration-200 cursor-pointer animate-fade-in-up ${
                  activeChatId === chat.id 
                    ? "bg-primary/10 border border-primary/20 shadow-sm" 
                    : "hover:bg-accent/50 border border-transparent hover:border-accent/30"
                }`}
                style={{ animationDelay: `${Math.min(index * 0.05, 0.25)}s` }}
              >
                <button
                  onClick={() => onChatSelect(chat.id)}
                  className="flex-1 text-left px-3 py-2.5 text-sm truncate transition-colors duration-200"
                >
                  <span className={activeChatId === chat.id ? "text-primary font-medium" : ""}>
                    {chat.title}
                  </span>
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onChatDelete(chat.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-2 hover:text-destructive transition-all duration-200 hover:scale-110 active:scale-95 mr-1"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="p-4 relative">
        <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-border/30 to-transparent" />
        <Button 
          variant="ghost" 
          className="w-full justify-start group relative overflow-hidden transition-all duration-300 hover:bg-accent/50" 
          onClick={onSettingsClick}
        >
          <Settings className="mr-2 h-4 w-4 transition-transform duration-300 group-hover:rotate-45" />
          <span>设置</span>
        </Button>
      </div>
    </div>
  )
}
