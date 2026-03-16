import { useState, useEffect, useCallback, useRef } from "react"
import { Settings, Upload, FileText, FolderOpen, RefreshCw, Trash2, Plus, ChevronRight, ChevronDown, MoreVertical, Brain, Save, Loader2, RefreshCcw, Check, AlertCircle, Sparkles, Clock, XCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Dialog, DialogContent } from "@/components/ui/dialog"
import { api } from "@/lib/api"
import type { Document, Category, GroupedDocuments } from "@/types"

interface SettingsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  documents: Document[]
  categories: Category[]
  onDocumentsChange: () => void
  onCategoriesChange: () => void
  defaultTab?: TabType
}

type TabType = "documents" | "models" | "rag" | "system"

interface ModelSettings {
  llm: {
    provider: string
    base_url: string
    model_name: string
    api_key: string
  }
  embedding: {
    provider: string
    base_url: string
    model_name: string
    dimension: number
  }
}

const LLM_PROVIDERS = [
  { value: "openai", label: "OpenAI", baseUrl: "https://api.openai.com/v1" },
  { value: "ollama", label: "Ollama", baseUrl: "http://localhost:11434" },
  { value: "lmstudio", label: "LM Studio", baseUrl: "http://localhost:1234/v1" },
  { value: "anthropic", label: "Anthropic", baseUrl: "https://api.anthropic.com" },
  { value: "azure", label: "Azure OpenAI", baseUrl: "" },
  { value: "custom", label: "Custom", baseUrl: "" },
]

const EMBEDDING_PROVIDERS = [
  { value: "huggingface", label: "HuggingFace", baseUrl: "" },
  { value: "ollama", label: "Ollama", baseUrl: "http://localhost:11434" },
  { value: "lmstudio", label: "LM Studio", baseUrl: "http://localhost:1234/v1" },
  { value: "openai", label: "OpenAI", baseUrl: "https://api.openai.com/v1" },
  { value: "custom", label: "Custom", baseUrl: "" },
]

const STATUS_CONFIG: Record<string, { icon: typeof Clock; color: string; bg: string; label: string; animate?: boolean }> = {
  pending: { icon: Clock, color: "text-yellow-500", bg: "bg-yellow-500/10", label: "等待处理" },
  processing: { icon: Loader2, color: "text-blue-500", bg: "bg-blue-500/10", label: "处理中", animate: true },
  completed: { icon: Check, color: "text-green-500", bg: "bg-green-500/10", label: "已完成" },
  failed: { icon: XCircle, color: "text-red-500", bg: "bg-red-500/10", label: "处理失败" },
}

export function SettingsDialog({ 
  open, 
  onOpenChange, 
  documents, 
  categories,
  onDocumentsChange,
  onCategoriesChange,
  defaultTab = "documents"
}: SettingsDialogProps) {
  const [activeTab, setActiveTab] = useState<TabType>(defaultTab)
  const [uploading, setUploading] = useState(false)
  const [reindexing, setReindexing] = useState(false)
  const [selectedCategoryId, setSelectedCategoryId] = useState<number | null>(null)
  const [expandedCategories, setExpandedCategories] = useState<Set<number>>(new Set())
  const [groupedDocuments, setGroupedDocuments] = useState<GroupedDocuments[]>([])
  const [newCategoryName, setNewCategoryName] = useState("")
  const [newCategoryColor, setNewCategoryColor] = useState("#6366f1")
  const [showNewCategoryInput, setShowNewCategoryInput] = useState(false)
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; categoryId: number } | null>(null)
  const contextMenuRef = useRef<HTMLDivElement>(null)
  const refreshIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  
  const [modelSettings, setModelSettings] = useState<ModelSettings>({
    llm: { provider: "openai", base_url: "https://api.openai.com/v1", model_name: "gpt-3.5-turbo", api_key: "" },
    embedding: { provider: "huggingface", base_url: "", model_name: "sentence-transformers/all-MiniLM-L6-v2", dimension: 384 }
  })
  const [savingModels, setSavingModels] = useState(false)
  const [modelsLoaded, setModelsLoaded] = useState(false)
  const [availableLlmModels, setAvailableLlmModels] = useState<string[]>([])
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState<string[]>([])
  const [loadingModels, setLoadingModels] = useState(false)
  const [modelLoadError, setModelLoadError] = useState<string>("")
  const [testingConnection, setTestingConnection] = useState(false)
  const [connectionResult, setConnectionResult] = useState<{ success: boolean; message: string } | null>(null)
  const [saveSuccess, setSaveSuccess] = useState(false)
  const [ragSettings, setRagSettings] = useState({
    parentDocument: { enabled: true, parentChunkSize: 2000, childChunkSize: 400 },
    compression: { enabled: true, threshold: 0.5, maxContextChars: 4000 }
  })
  const [ragSettingsLoaded, setRagSettingsLoaded] = useState(false)

  const hasProcessingDocs = groupedDocuments.some(g => 
    g.documents.some(d => d.status === "pending" || d.status === "processing")
  )

  useEffect(() => {
    if (open && activeTab === "documents" && hasProcessingDocs) {
      refreshIntervalRef.current = setInterval(() => {
        loadGroupedDocuments()
      }, 3000)
    }
    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current)
        refreshIntervalRef.current = null
      }
    }
  }, [open, activeTab, hasProcessingDocs])

  const loadAvailableModels = useCallback(async (provider: string, baseUrl: string, type: "llm" | "embedding") => {
    setModelLoadError("")
    
    if (!baseUrl || (provider !== "ollama" && provider !== "lmstudio")) {
      if (type === "llm") setAvailableLlmModels([])
      else setAvailableEmbeddingModels([])
      return
    }

    setLoadingModels(true)
    try {
      const response = await api.models.getAvailable(provider, baseUrl)
      setLoadingModels(false)

      if (response.error) {
        setModelLoadError(`获取模型失败: ${response.error}`)
        if (type === "llm") setAvailableLlmModels([])
        else setAvailableEmbeddingModels([])
        return
      }

      if (response.data?.models && response.data.models.length > 0) {
        const models = response.data.models
        if (type === "llm") {
          setAvailableLlmModels(models)
          if (!models.includes(modelSettings.llm.model_name)) {
            setModelSettings(prev => ({
              ...prev,
              llm: { ...prev.llm, model_name: models[0] }
            }))
          }
        } else {
          setAvailableEmbeddingModels(models)
          if (!models.includes(modelSettings.embedding.model_name)) {
            setModelSettings(prev => ({
              ...prev,
              embedding: { ...prev.embedding, model_name: models[0] }
            }))
          }
        }
      } else if (response.data?.error) {
        setModelLoadError(`获取模型失败: ${response.data.error}`)
        if (type === "llm") setAvailableLlmModels([])
        else setAvailableEmbeddingModels([])
      } else {
        setModelLoadError("未找到已加载的模型，请确保 LM Studio 或 Ollama 已启动并加载了模型")
        if (type === "llm") setAvailableLlmModels([])
        else setAvailableEmbeddingModels([])
      }
    } catch (error) {
      setLoadingModels(false)
      setModelLoadError(`请求失败: ${error instanceof Error ? error.message : "未知错误"}`)
      if (type === "llm") setAvailableLlmModels([])
      else setAvailableEmbeddingModels([])
    }
  }, [modelSettings.llm.model_name, modelSettings.embedding.model_name])

  useEffect(() => {
    if (open && !modelsLoaded) {
      loadModelSettings()
    }
  }, [open])

  useEffect(() => {
    if (open) {
      setActiveTab(defaultTab)
    }
  }, [open, defaultTab])

  useEffect(() => {
    if (open && activeTab === "documents") {
      loadGroupedDocuments()
    }
  }, [open, activeTab, documents])

  useEffect(() => {
    if (open && activeTab === "rag" && !ragSettingsLoaded) {
      loadRagSettings()
    }
  }, [open, activeTab])

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (contextMenuRef.current && !contextMenuRef.current.contains(e.target as Node)) {
        setContextMenu(null)
      }
    }
    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [])

  const loadGroupedDocuments = async () => {
    const response = await api.documents.grouped()
    if (response.data) {
      setGroupedDocuments(response.data)
      const allCategoryIds = response.data
        .filter((g: GroupedDocuments) => g.category !== null)
        .map((g: GroupedDocuments) => g.category!.id)
      setExpandedCategories(new Set(allCategoryIds))
    }
  }

  const loadModelSettings = async () => {
    const response = await api.settings.getModels()
    if (response.data) {
      setModelSettings(response.data)
      setModelsLoaded(true)
    }
  }

  const loadRagSettings = async () => {
    try {
      const response = await api.settings.getRagSettings()
      if (response.data) {
        setRagSettings(response.data)
        setRagSettingsLoaded(true)
      }
    } catch (error) {
      console.error("Failed to load RAG settings:", error)
    }
  }

  const handleSaveRagSettings = async () => {
    setSavingModels(true)
    setSaveSuccess(false)
    try {
      await api.settings.updateRagSettings(ragSettings)
      setSaveSuccess(true)
      setTimeout(() => setSaveSuccess(false), 3000)
    } catch (error) {
      alert("保存失败：" + error)
    }
    setSavingModels(false)
  }

  const handleSaveModelSettings = async () => {
    setSavingModels(true)
    setSaveSuccess(false)
    try {
      await api.settings.updateModels(modelSettings)
      setSaveSuccess(true)
      setTimeout(() => setSaveSuccess(false), 3000)
    } catch (error) {
      alert("保存失败：" + error)
    }
    setSavingModels(false)
  }

  const handleTestConnection = async (type: "llm" | "embedding") => {
    setTestingConnection(true)
    setConnectionResult(null)
    
    try {
      const settings = type === "llm" ? modelSettings.llm : modelSettings.embedding
      const response = await api.models.testConnection(
        settings.provider,
        settings.base_url,
        settings.model_name,
        type === "llm" ? modelSettings.llm.api_key : ""
      )
      
      if (response.data) {
        setConnectionResult({
          success: response.data.success,
          message: response.data.success ? response.data.message || "连接成功" : response.data.error || "连接失败"
        })
      } else if (response.error) {
        setConnectionResult({
          success: false,
          message: response.error
        })
      }
    } catch (error) {
      setConnectionResult({
        success: false,
        message: error instanceof Error ? error.message : "未知错误"
      })
    }
    
    setTestingConnection(false)
  }

  const handleUpload = async () => {
    const input = document.createElement("input")
    input.type = "file"
    input.accept = ".txt,.md,.pdf"
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return
      
      setUploading(true)
      try {
        const result = await api.documents.upload(file, selectedCategoryId || undefined)
        if (result.error) {
          alert(`上传失败: ${result.error}`)
        } else {
          loadGroupedDocuments()
        }
      } catch (error) {
        alert("上传失败")
      }
      setUploading(false)
      onDocumentsChange()
    }
    input.click()
  }

  const handleReindex = async () => {
    setReindexing(true)
    try {
      await api.documents.reindex()
      loadGroupedDocuments()
    } catch (error) {
      alert("重新索引失败")
    }
    setReindexing(false)
  }

  const handleDeleteDocument = async (id: number) => {
    if (!confirm("确定要删除这个文档吗？")) return
    await api.documents.delete(id)
    onDocumentsChange()
    loadGroupedDocuments()
  }

  const toggleCategory = (categoryId: number) => {
    setExpandedCategories(prev => {
      const newSet = new Set(prev)
      if (newSet.has(categoryId)) {
        newSet.delete(categoryId)
      } else {
        newSet.add(categoryId)
      }
      return newSet
    })
  }

  const handleCreateCategory = async () => {
    if (!newCategoryName.trim()) return
    const result = await api.categories.create({ name: newCategoryName, color: newCategoryColor })
    if (result.error) {
      alert(`创建分类失败: ${result.error}`)
    } else {
      setNewCategoryName("")
      setShowNewCategoryInput(false)
      onCategoriesChange()
      loadGroupedDocuments()
    }
  }

  const handleDeleteCategory = async (categoryId: number) => {
    const category = categories.find(c => c.id === categoryId)
    if (category?.name === "Default") {
      alert("默认分类不能删除")
      return
    }
    if (!confirm(`确定要删除分类 "${category?.name}" 吗？该分类下的文档将移动到 Default 分类。`)) return
    
    const result = await api.categories.delete(categoryId)
    if (result.error) {
      alert(`删除分类失败: ${result.error}`)
    } else {
      onCategoriesChange()
      loadGroupedDocuments()
    }
    setContextMenu(null)
  }

  const handleContextMenu = (e: React.MouseEvent, categoryId: number) => {
    e.preventDefault()
    setContextMenu({ x: e.clientX, y: e.clientY, categoryId })
  }

  const renderStatusBadge = (status: Document["status"]) => {
    const config = STATUS_CONFIG[status]
    const Icon = config.icon
    return (
      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs ${config.bg} ${config.color}`}>
        <Icon className={`h-3 w-3 ${config.animate ? "animate-spin" : ""}`} />
        {config.label}
      </span>
    )
  }

  const tabs = [
    { id: "documents" as TabType, label: "文档管理", icon: FileText },
    { id: "models" as TabType, label: "模型设置", icon: Brain },
    { id: "rag" as TabType, label: "RAG 增强", icon: Sparkles },
    { id: "system" as TabType, label: "系统设置", icon: Settings },
  ]

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl h-[700px] p-0 flex animate-scale-in">
        <div className="w-48 bg-gradient-to-b from-muted/50 to-muted/30 p-4 relative">
          <div className="absolute right-0 top-4 bottom-4 w-px bg-gradient-to-b from-transparent via-border/40 to-transparent" />
          <div className="flex items-center gap-2 mb-6">
            <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
              <Settings className="h-4 w-4 text-primary" />
            </div>
            <h2 className="text-lg font-semibold">设置</h2>
          </div>
          <nav className="space-y-1">
            {tabs.map((tab, idx) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-300 animate-fade-in-up ${
                  activeTab === tab.id 
                    ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20" 
                    : "hover:bg-accent/50 text-muted-foreground hover:text-foreground"
                }`}
                style={{ animationDelay: `${idx * 0.05}s` }}
              >
                <tab.icon className="h-4 w-4" />
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        <div className="flex-1 flex flex-col">
          <div className="p-4 flex items-center justify-between bg-card/50 relative">
            <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-border/30 to-transparent" />
            <h3 className="font-semibold flex items-center gap-2">
              {tabs.find(t => t.id === activeTab)?.icon && (
                <span className="w-6 h-6 rounded bg-primary/10 flex items-center justify-center">
                  {(() => {
                    const TabIcon = tabs.find(t => t.id === activeTab)?.icon
                    return TabIcon && <TabIcon className="h-3.5 w-3.5 text-primary" />
                  })()}
                </span>
              )}
              {tabs.find(t => t.id === activeTab)?.label}
            </h3>
            {hasProcessingDocs && (
              <span className="text-xs text-blue-500 flex items-center gap-1 animate-pulse">
                <Loader2 className="h-3 w-3 animate-spin" />
                正在处理文档...
              </span>
            )}
          </div>

          <div className="flex-1 overflow-auto p-4 scrollbar-thin">
            {activeTab === "documents" && (
              <div className="space-y-4 animate-fade-in">
                <div className="flex gap-3 items-center flex-wrap">
                  <Button 
                    onClick={handleUpload} 
                    disabled={uploading}
                    className="transition-all duration-300 hover:shadow-lg hover:shadow-primary/20 hover:scale-[1.02] active:scale-[0.98]"
                  >
                    {uploading ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Upload className="h-4 w-4 mr-2" />
                    )}
                    {uploading ? "上传中..." : "上传文档"}
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={handleReindex} 
                    disabled={reindexing}
                    className="transition-all duration-300 hover:bg-accent hover:border-violet-500/50"
                  >
                    <RefreshCw className={`h-4 w-4 mr-2 ${reindexing ? "animate-spin" : ""}`} />
                    {reindexing ? "索引中..." : "重新索引"}
                  </Button>
                  <div className="flex items-center gap-2 ml-auto">
                    <span className="text-sm text-muted-foreground">上传到分类:</span>
                    <select
                      value={selectedCategoryId || ""}
                      onChange={(e) => setSelectedCategoryId(e.target.value ? Number(e.target.value) : null)}
                      className="px-3 py-1.5 border border-violet-500/30 rounded-lg bg-background text-sm transition-all duration-200 focus:ring-2 focus:ring-violet-500/30 focus:border-violet-500/50"
                    >
                      <option value="">Default</option>
                      {categories.filter(c => c.name !== "Default").map(c => (
                        <option key={c.id} value={c.id}>{c.name}</option>
                      ))}
                    </select>
                  </div>
                </div>

                <div className="relative rounded-2xl p-[1px] bg-gradient-to-br from-violet-500/20 via-purple-500/10 to-fuchsia-500/20">
                  <div className="rounded-[13px] overflow-hidden shadow-sm bg-gradient-to-br from-card to-card/80 backdrop-blur-sm">
                    <div className="p-3 bg-muted/50 flex items-center justify-between relative">
                      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-border/30 to-transparent" />
                      <span className="text-sm font-medium">文档分类</span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowNewCategoryInput(true)}
                      className="h-7 text-xs transition-all duration-200 hover:bg-primary/10 hover:text-primary"
                    >
                      <Plus className="h-3 w-3 mr-1" />
                      新建分类
                    </Button>
                  </div>
                  
                  {showNewCategoryInput && (
                    <div className="p-3 bg-accent/30 flex items-center gap-2 animate-fade-in relative">
                      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-border/20 to-transparent" />
                      <Input
                        placeholder="分类名称"
                        value={newCategoryName}
                        onChange={(e) => setNewCategoryName(e.target.value)}
                        className="h-8 w-40 text-sm"
                        autoFocus
                      />
                      <Input
                        type="color"
                        value={newCategoryColor}
                        onChange={(e) => setNewCategoryColor(e.target.value)}
                        className="w-8 h-8 p-0.5 cursor-pointer"
                      />
                      <Button size="sm" onClick={handleCreateCategory} className="h-8">
                        确定
                      </Button>
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        onClick={() => { setShowNewCategoryInput(false); setNewCategoryName("") }}
                        className="h-8"
                      >
                        取消
                      </Button>
                    </div>
                  )}

                  <div className="divide-y max-h-[400px] overflow-auto scrollbar-thin">
                    {groupedDocuments.map((group, idx) => (
                      <div key={group.category?.id || 'uncategorized'} className="animate-fade-in" style={{ animationDelay: `${idx * 0.05}s` }}>
                        <div
                          className={`flex items-center gap-2 px-3 py-2.5 cursor-pointer transition-all duration-200 hover:bg-accent/50 ${
                            group.category?.name === "Default" ? "" : "group"
                          }`}
                          onClick={() => group.category && toggleCategory(group.category.id)}
                          onContextMenu={(e) => group.category && group.category.name !== "Default" && handleContextMenu(e, group.category.id)}
                        >
                          {group.category && expandedCategories.has(group.category.id) ? (
                            <ChevronDown className="h-4 w-4 text-muted-foreground transition-transform duration-200" />
                          ) : group.category ? (
                            <ChevronRight className="h-4 w-4 text-muted-foreground transition-transform duration-200" />
                          ) : (
                            <div className="w-4" />
                          )}
                          <div
                            className="w-3 h-3 rounded-full shrink-0"
                            style={{ backgroundColor: group.category?.color || "#94a3b8" }}
                          />
                          <FolderOpen className="h-4 w-4 text-primary" />
                          <span className="flex-1 text-sm font-medium truncate">
                            {group.category?.name || "未分类"}
                          </span>
                          <span className="text-xs text-muted-foreground bg-muted px-2 py-0.5 rounded-full">
                            {group.documents.length}
                          </span>
                          {group.category && group.category.name !== "Default" && (
                            <MoreVertical className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
                          )}
                        </div>
                        
                        {group.category && expandedCategories.has(group.category.id) && group.documents.length > 0 && (
                          <div className="pl-10 pr-3 pb-2 space-y-1 animate-fade-in">
                            {group.documents.map((doc, docIdx) => (
                              <div
                                key={doc.id}
                                className="flex items-center gap-2 px-2 py-2 rounded-lg hover:bg-accent/30 transition-all duration-200 group/doc animate-fade-in-up"
                                style={{ animationDelay: `${docIdx * 0.03}s` }}
                              >
                                <FileText className="h-4 w-4 text-muted-foreground shrink-0" />
                                <span className="flex-1 text-sm truncate">{doc.title}</span>
                                {renderStatusBadge(doc.status)}
                                <span className="text-xs text-muted-foreground uppercase">{doc.file_type}</span>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => handleDeleteDocument(doc.id)}
                                  className="h-6 w-6 p-0 opacity-0 group-hover/doc:opacity-100 hover:text-destructive hover:bg-destructive/10 transition-all duration-200"
                                >
                                  <Trash2 className="h-3 w-3" />
                                </Button>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                    
                    {groupedDocuments.length === 0 && (
                      <div className="p-12 text-center">
                        <div className="flex flex-col items-center gap-3 text-muted-foreground animate-fade-in">
                          <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center">
                            <FileText className="h-8 w-8 opacity-50" />
                          </div>
                          <p className="text-sm">暂无文档，请上传文档开始使用</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
                </div>
              </div>
            )}

            {activeTab === "models" && (
              <div className="space-y-6 animate-fade-in">
                <div className="relative rounded-2xl p-[1px] bg-gradient-to-br from-blue-500/20 via-cyan-500/10 to-teal-500/20">
                  <div className="rounded-[13px] p-5 bg-gradient-to-br from-card to-card/80 backdrop-blur-sm">
                    <h4 className="font-medium mb-4 flex items-center gap-2">
                      <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                        <Brain className="h-4 w-4 text-primary" />
                      </div>
                      LLM（大语言模型）设置
                    </h4>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-1.5">提供商</label>
                      <div className="flex gap-2">
                        <select
                          value={modelSettings.llm.provider}
                          onChange={(e) => {
                            const provider = LLM_PROVIDERS.find(p => p.value === e.target.value)
                            setModelSettings({
                              ...modelSettings,
                              llm: { 
                                ...modelSettings.llm, 
                                provider: e.target.value,
                                base_url: provider?.baseUrl || modelSettings.llm.base_url
                              }
                            })
                            loadAvailableModels(e.target.value, provider?.baseUrl || "", "llm")
                          }}
                          className="flex-1 p-2.5 border border-blue-500/30 rounded-lg bg-background transition-all duration-200 focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500/50"
                        >
                          {LLM_PROVIDERS.map(p => (
                            <option key={p.value} value={p.value}>{p.label}</option>
                          ))}
                        </select>
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => loadAvailableModels(modelSettings.llm.provider, modelSettings.llm.base_url, "llm")}
                          disabled={loadingModels || (modelSettings.llm.provider !== "ollama" && modelSettings.llm.provider !== "lmstudio")}
                          className="transition-all duration-200 hover:bg-primary/5"
                        >
                          <RefreshCcw className={`h-4 w-4 ${loadingModels ? "animate-spin" : ""}`} />
                        </Button>
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1.5">Base URL</label>
                      <Input
                        value={modelSettings.llm.base_url}
                        onChange={(e) => setModelSettings({
                          ...modelSettings,
                          llm: { ...modelSettings.llm, base_url: e.target.value }
                        })}
                        placeholder="https://api.openai.com/v1"
                        className="transition-all duration-300 focus:ring-2 focus:ring-primary/30"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1.5">模型名称</label>
                      {(modelSettings.llm.provider === "ollama" || modelSettings.llm.provider === "lmstudio") && availableLlmModels.length > 0 ? (
                        <select
                          value={modelSettings.llm.model_name}
                          onChange={(e) => setModelSettings({
                            ...modelSettings,
                            llm: { ...modelSettings.llm, model_name: e.target.value }
                          })}
                          className="w-full p-2.5 border border-blue-500/30 rounded-lg bg-background transition-all duration-200 focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500/50"
                        >
                          {availableLlmModels.map(m => (
                            <option key={m} value={m}>{m}</option>
                          ))}
                        </select>
                      ) : (
                        <Input
                          value={modelSettings.llm.model_name}
                          onChange={(e) => setModelSettings({
                            ...modelSettings,
                            llm: { ...modelSettings.llm, model_name: e.target.value }
                          })}
                          placeholder="gpt-3.5-turbo"
                          className="transition-all duration-300 focus:ring-2 focus:ring-primary/30"
                        />
                      )}
                      {modelLoadError && (modelSettings.llm.provider === "ollama" || modelSettings.llm.provider === "lmstudio") && (
                        <p className="text-sm text-destructive mt-2 flex items-center gap-1">
                          <AlertCircle className="h-3.5 w-3.5" />
                          {modelLoadError}
                        </p>
                      )}
                    </div>
                    {modelSettings.llm.provider !== "ollama" && modelSettings.llm.provider !== "lmstudio" && (
                    <div>
                      <label className="block text-sm font-medium mb-1.5">API Key</label>
                      <Input
                        type="password"
                        value={modelSettings.llm.api_key}
                        onChange={(e) => setModelSettings({
                          ...modelSettings,
                          llm: { ...modelSettings.llm, api_key: e.target.value }
                        })}
                        placeholder="sk-..."
                        className="transition-all duration-300 focus:ring-2 focus:ring-primary/30"
                      />
                    </div>
                    )}
                    <div className="flex items-center gap-3">
                      <Button 
                        variant="outline" 
                        onClick={() => handleTestConnection("llm")}
                        disabled={testingConnection || !modelSettings.llm.base_url}
                        className="transition-all duration-300 hover:bg-primary/5"
                      >
                        {testingConnection ? (
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        ) : null}
                        测试连接
                      </Button>
                      {connectionResult && (
                        <span className={`text-sm flex items-center gap-1.5 animate-fade-in ${
                          connectionResult.success ? "text-green-600" : "text-destructive"
                        }`}>
                          {connectionResult.success ? (
                            <Check className="h-4 w-4" />
                          ) : (
                            <AlertCircle className="h-4 w-4" />
                          )}
                          {connectionResult.message}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                </div>

                <div className="relative rounded-2xl p-[1px] bg-gradient-to-br from-emerald-500/20 via-teal-500/10 to-cyan-500/20">
                  <div className="rounded-[13px] p-5 bg-gradient-to-br from-card to-card/80 backdrop-blur-sm">
                    <h4 className="font-medium mb-4 flex items-center gap-2">
                      <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                        <Brain className="h-4 w-4 text-primary" />
                      </div>
                      Embedding（向量化模型）设置
                    </h4>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-1.5">提供商</label>
                      <div className="flex gap-2">
                        <select
                          value={modelSettings.embedding.provider}
                          onChange={(e) => {
                            const provider = EMBEDDING_PROVIDERS.find(p => p.value === e.target.value)
                            setModelSettings({
                              ...modelSettings,
                              embedding: { 
                                ...modelSettings.embedding, 
                                provider: e.target.value,
                                base_url: provider?.baseUrl || modelSettings.embedding.base_url
                              }
                            })
                            if (provider?.baseUrl) {
                              loadAvailableModels(e.target.value, provider.baseUrl, "embedding")
                            }
                          }}
                          className="flex-1 p-2.5 border border-emerald-500/30 rounded-lg bg-background transition-all duration-200 focus:ring-2 focus:ring-emerald-500/30 focus:border-emerald-500/50"
                        >
                          {EMBEDDING_PROVIDERS.map(p => (
                            <option key={p.value} value={p.value}>{p.label}</option>
                          ))}
                        </select>
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => loadAvailableModels(modelSettings.embedding.provider, modelSettings.embedding.base_url, "embedding")}
                          disabled={loadingModels || (modelSettings.embedding.provider !== "ollama" && modelSettings.embedding.provider !== "lmstudio")}
                          className="transition-all duration-200 hover:bg-primary/5"
                        >
                          <RefreshCcw className={`h-4 w-4 ${loadingModels ? "animate-spin" : ""}`} />
                        </Button>
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1.5">Base URL</label>
                      <Input
                        value={modelSettings.embedding.base_url}
                        onChange={(e) => setModelSettings({
                          ...modelSettings,
                          embedding: { ...modelSettings.embedding, base_url: e.target.value }
                        })}
                        placeholder="http://localhost:11434"
                        className="transition-all duration-300 focus:ring-2 focus:ring-primary/30"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1.5">模型名称</label>
                      {(modelSettings.embedding.provider === "ollama" || modelSettings.embedding.provider === "lmstudio") && availableEmbeddingModels.length > 0 ? (
                        <select
                          value={modelSettings.embedding.model_name}
                          onChange={(e) => setModelSettings({
                            ...modelSettings,
                            embedding: { ...modelSettings.embedding, model_name: e.target.value }
                          })}
                          className="w-full p-2.5 border border-emerald-500/30 rounded-lg bg-background transition-all duration-200 focus:ring-2 focus:ring-emerald-500/30 focus:border-emerald-500/50"
                        >
                          {availableEmbeddingModels.map(m => (
                            <option key={m} value={m}>{m}</option>
                          ))}
                        </select>
                      ) : (
                        <Input
                          value={modelSettings.embedding.model_name}
                          onChange={(e) => setModelSettings({
                            ...modelSettings,
                            embedding: { ...modelSettings.embedding, model_name: e.target.value }
                          })}
                          placeholder="sentence-transformers/all-MiniLM-L6-v2"
                          className="transition-all duration-300 focus:ring-2 focus:ring-primary/30"
                        />
                      )}
                    </div>
                    <div className="flex items-center gap-3">
                      <Button 
                        variant="outline" 
                        onClick={() => handleTestConnection("embedding")}
                        disabled={testingConnection || (modelSettings.embedding.provider !== "huggingface" && !modelSettings.embedding.base_url)}
                        className="transition-all duration-300 hover:bg-primary/5"
                      >
                        {testingConnection ? (
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        ) : null}
                        测试连接
                      </Button>
                      {connectionResult && (
                        <span className={`text-sm flex items-center gap-1.5 animate-fade-in ${
                          connectionResult.success ? "text-green-600" : "text-destructive"
                        }`}>
                          {connectionResult.success ? (
                            <Check className="h-4 w-4" />
                          ) : (
                            <AlertCircle className="h-4 w-4" />
                          )}
                          {connectionResult.message}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                </div>

                <Button 
                  onClick={handleSaveModelSettings} 
                  disabled={savingModels}
                  className={`transition-all duration-300 hover:shadow-lg hover:shadow-primary/20 hover:scale-[1.02] active:scale-[0.98] ${
                    saveSuccess ? "bg-green-600 hover:bg-green-700" : ""
                  }`}
                >
                  {savingModels ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : saveSuccess ? (
                    <Check className="h-4 w-4 mr-2" />
                  ) : (
                    <Save className="h-4 w-4 mr-2" />
                  )}
                  {saveSuccess ? "已保存" : "保存设置"}
                </Button>
              </div>
            )}

            {activeTab === "rag" && (
              <div className="space-y-6 animate-fade-in">
                <div className="relative rounded-2xl p-[1px] bg-gradient-to-br from-violet-500/20 via-purple-500/10 to-fuchsia-500/20">
                  <div className="rounded-[13px] p-5 bg-gradient-to-br from-card to-card/80 backdrop-blur-sm">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center shadow-lg shadow-violet-500/25">
                        <Sparkles className="h-5 w-5 text-white" />
                      </div>
                      <div>
                        <h4 className="font-semibold">RAG 2.0 增强</h4>
                        <p className="text-xs text-muted-foreground">检索增强生成优化设置</p>
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground leading-relaxed mb-4">
                      RAG 2.0 提供先进的检索技术，提升问答系统的准确性和效率。
                    </p>
                  </div>
                </div>

                <div className="relative rounded-2xl p-[1px] bg-gradient-to-br from-violet-500/20 via-purple-500/10 to-fuchsia-500/20">
                  <div className="rounded-[13px] p-5 bg-gradient-to-br from-card to-card/80 backdrop-blur-sm">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h4 className="font-medium">父文档检索 (Parent Document Retriever)</h4>
                        <p className="text-xs text-muted-foreground mt-1">先检索小片段，再获取完整父文档</p>
                      </div>
                      <button
                        onClick={() => setRagSettings(prev => ({
                          ...prev,
                          parentDocument: { ...prev.parentDocument, enabled: !prev.parentDocument.enabled }
                        }))}
                        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 shrink-0 ${
                          ragSettings.parentDocument.enabled ? "bg-violet-500" : "bg-gray-300"
                        }`}
                      >
                        <span
                          className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform duration-200 ${
                            ragSettings.parentDocument.enabled ? "translate-x-6" : "translate-x-1"
                          }`}
                        />
                      </button>
                    </div>
                    {ragSettings.parentDocument.enabled && (
                      <div className="grid grid-cols-2 gap-4 mt-4 pt-4 border-t border-violet-500/20">
                        <div className="space-y-2">
                          <label className="text-xs text-muted-foreground">父文档大小</label>
                          <Input
                            type="number"
                            value={ragSettings.parentDocument.parentChunkSize}
                            onChange={(e) => setRagSettings(prev => ({
                              ...prev,
                              parentDocument: { ...prev.parentDocument, parentChunkSize: parseInt(e.target.value) || 2000 }
                            }))}
                            min={500}
                            max={5000}
                            step={100}
                          />
                          <p className="text-xs text-muted-foreground">较大值提供更完整上下文</p>
                        </div>
                        <div className="space-y-2">
                          <label className="text-xs text-muted-foreground">子块大小</label>
                          <Input
                            type="number"
                            value={ragSettings.parentDocument.childChunkSize}
                            onChange={(e) => setRagSettings(prev => ({
                              ...prev,
                              parentDocument: { ...prev.parentDocument, childChunkSize: parseInt(e.target.value) || 400 }
                            }))}
                            min={100}
                            max={1000}
                            step={50}
                          />
                          <p className="text-xs text-muted-foreground">用于向量检索的片段大小</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                <div className="relative rounded-2xl p-[1px] bg-gradient-to-br from-violet-500/20 via-purple-500/10 to-fuchsia-500/20">
                  <div className="rounded-[13px] p-5 bg-gradient-to-br from-card to-card/80 backdrop-blur-sm">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h4 className="font-medium">上下文压缩 (Contextual Compression)</h4>
                        <p className="text-xs text-muted-foreground mt-1">过滤低相关性文档，限制上下文长度</p>
                      </div>
                      <button
                        onClick={() => setRagSettings(prev => ({
                          ...prev,
                          compression: { ...prev.compression, enabled: !prev.compression.enabled }
                        }))}
                        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 shrink-0 ${
                          ragSettings.compression.enabled ? "bg-violet-500" : "bg-gray-300"
                        }`}
                      >
                        <span
                          className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform duration-200 ${
                            ragSettings.compression.enabled ? "translate-x-6" : "translate-x-1"
                          }`}
                        />
                      </button>
                    </div>
                    {ragSettings.compression.enabled && (
                      <div className="grid grid-cols-2 gap-4 mt-4 pt-4 border-t border-violet-500/20">
                        <div className="space-y-2">
                          <label className="text-xs text-muted-foreground">相似度阈值</label>
                          <Input
                            type="number"
                            value={ragSettings.compression.threshold}
                            onChange={(e) => setRagSettings(prev => ({
                              ...prev,
                              compression: { ...prev.compression, threshold: parseFloat(e.target.value) || 0.5 }
                            }))}
                            min={0}
                            max={1}
                            step={0.1}
                          />
                          <p className="text-xs text-muted-foreground">过滤低于此相似度的文档</p>
                        </div>
                        <div className="space-y-2">
                          <label className="text-xs text-muted-foreground">最大字符数</label>
                          <Input
                            type="number"
                            value={ragSettings.compression.maxContextChars}
                            onChange={(e) => setRagSettings(prev => ({
                              ...prev,
                              compression: { ...prev.compression, maxContextChars: parseInt(e.target.value) || 4000 }
                            }))}
                            min={1000}
                            max={10000}
                            step={500}
                          />
                          <p className="text-xs text-muted-foreground">发送给 LLM 的最大字符数</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                <div className="relative rounded-2xl p-[1px] bg-gradient-to-br from-amber-500/20 via-orange-500/10 to-yellow-500/20">
                  <div className="rounded-[13px] p-4 bg-gradient-to-br from-amber-500/10 to-orange-500/10 backdrop-blur-sm">
                    <div className="flex items-start gap-3">
                      <AlertCircle className="h-5 w-5 text-amber-500 mt-0.5" />
                      <div className="text-sm">
                        <p className="font-medium text-amber-700">注意事项</p>
                        <p className="text-xs text-amber-600/80 mt-1">
                          修改父文档设置后，需要重新上传文档才能生效。压缩设置即时生效。
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="flex justify-end">
                  <Button 
                    onClick={handleSaveRagSettings} 
                    disabled={savingModels}
                    className={`transition-all duration-300 hover:shadow-lg hover:shadow-violet-500/20 hover:scale-[1.02] active:scale-[0.98] ${
                      saveSuccess ? "bg-green-600 hover:bg-green-700" : ""
                    }`}
                  >
                    {savingModels ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : saveSuccess ? (
                      <Check className="h-4 w-4 mr-2" />
                    ) : (
                      <Save className="h-4 w-4 mr-2" />
                    )}
                    {saveSuccess ? "已保存" : "保存设置"}
                  </Button>
                </div>
              </div>
            )}

            {activeTab === "system" && (
              <div className="space-y-6 animate-fade-in">
                <div className="relative rounded-2xl p-[1px] bg-gradient-to-br from-primary/20 via-primary/10 to-primary/20">
                  <div className="rounded-[13px] p-5 bg-gradient-to-br from-card to-card/80 backdrop-blur-sm">
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-primary/70 flex items-center justify-center shadow-lg shadow-primary/20">
                        <Sparkles className="h-5 w-5 text-primary-foreground" />
                      </div>
                      <div>
                        <h4 className="font-semibold">LocalBrain</h4>
                        <p className="text-xs text-muted-foreground">本地知识库管理系统</p>
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      LocalBrain 是一个本地知识库管理系统，支持文档上传、语义搜索和智能问答。
                      所有数据都存储在本地，保护你的隐私安全。
                    </p>
                  </div>
                </div>
                <div className="relative rounded-2xl p-[1px] bg-gradient-to-br from-primary/20 via-primary/10 to-primary/20">
                  <div className="rounded-[13px] p-5 bg-gradient-to-br from-card to-card/80 backdrop-blur-sm">
                    <h4 className="font-medium mb-3">版本信息</h4>
                    <div className="flex items-center gap-2">
                      <span className="px-3 py-1 rounded-full bg-primary/10 text-primary text-sm font-medium">
                        v1.0.0
                      </span>
                    </div>
                  </div>
                </div>
                <div className="relative rounded-2xl p-[1px] bg-gradient-to-br from-primary/20 via-primary/10 to-primary/20">
                  <div className="rounded-[13px] p-5 bg-gradient-to-br from-card to-card/80 backdrop-blur-sm">
                    <h4 className="font-medium mb-3">快捷键</h4>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">新建对话</span>
                        <kbd className="px-2 py-1 bg-muted rounded text-xs font-mono">Ctrl+N</kbd>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {contextMenu && (
          <div
            ref={contextMenuRef}
            className="fixed z-50 bg-card border border-violet-500/30 rounded-lg shadow-lg py-1 min-w-[120px] animate-scale-in"
            style={{ left: contextMenu.x, top: contextMenu.y }}
          >
            <button
              onClick={() => handleDeleteCategory(contextMenu.categoryId)}
              className="w-full px-3 py-2 text-left text-sm hover:bg-destructive/10 hover:text-destructive transition-colors duration-200 flex items-center gap-2"
            >
              <Trash2 className="h-4 w-4" />
              删除分类
            </button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
