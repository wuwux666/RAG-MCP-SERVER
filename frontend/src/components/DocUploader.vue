<script setup lang="ts">
import { ref } from 'vue'

const emit = defineEmits<{
  upload: [file: File]
}>()

const dragging = ref(false)
const uploading = ref(false)
const fileInput = ref<HTMLInputElement>()

function onDragOver(e: DragEvent) {
  e.preventDefault()
  dragging.value = true
}

function onDragLeave() {
  dragging.value = false
}

function onDrop(e: DragEvent) {
  e.preventDefault()
  dragging.value = false
  const file = e.dataTransfer?.files?.[0]
  if (file) handleFile(file)
}

function onFileChange(e: Event) {
  const target = e.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) handleFile(file)
}

async function handleFile(file: File) {
  uploading.value = true
  try {
    emit('upload', file)
  } finally {
    uploading.value = false
    if (fileInput.value) fileInput.value.value = ''
  }
}
</script>

<template>
  <div
    class="border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-300"
    :class="dragging
      ? 'border-sakura-pink bg-sakura-bg scale-[1.02]'
      : 'border-sakura-light/60 hover:border-sakura-pink hover:bg-sakura-bg/50'"
    @dragover="onDragOver"
    @dragleave="onDragLeave"
    @drop="onDrop"
    @click="fileInput?.click()"
  >
    <input
      ref="fileInput"
      type="file"
      accept=".pdf,.txt,.md,.docx"
      class="hidden"
      @change="onFileChange"
    />

    <div v-if="uploading" class="flex flex-col items-center gap-2">
      <div class="w-10 h-10 border-4 border-sakura-light border-t-sakura-pink rounded-full animate-spin"></div>
      <span class="text-sakura-muted font-medium">上传中...</span>
    </div>

    <div v-else class="flex flex-col items-center gap-3">
      <div class="text-4xl">📄</div>
      <div>
        <p class="text-sakura-text font-semibold text-lg">拖拽文件到此处，或点击选择</p>
        <p class="text-sakura-muted text-sm mt-1">支持 PDF、TXT、Markdown、DOCX 格式</p>
      </div>
    </div>
  </div>
</template>
