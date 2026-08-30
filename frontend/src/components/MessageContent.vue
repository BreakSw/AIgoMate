<script setup lang="ts">
import { computed } from 'vue'

type InlinePart = { text: string; kind: 'text' | 'strong' | 'code' | 'link'; href?: string }
type Block = {
  type: 'paragraph' | 'heading' | 'list' | 'code' | 'table' | 'rule'
  parts?: InlinePart[]
  items?: InlinePart[][]
  code?: string
  language?: string
  headers?: InlinePart[][]
  rows?: InlinePart[][][]
}

const props = withDefaults(defineProps<{ content: string; tone?: 'user' | 'assistant' | 'live' }>(), {
  tone: 'assistant',
})

function parseInline(text: string): InlinePart[] {
  const result: InlinePart[] = []
  const pattern = /(\*\*[^*]+\*\*|`[^`\n]+`|\[[^\]\n]+\]\(https?:\/\/[^\s)]+\))/g
  let cursor = 0
  for (const match of text.matchAll(pattern)) {
    const index = match.index ?? 0
    if (index > cursor) result.push({ text: text.slice(cursor, index), kind: 'text' })
    const token = match[0]
    if (token.startsWith('**')) result.push({ text: token.slice(2, -2), kind: 'strong' })
    else if (token.startsWith('`')) result.push({ text: token.slice(1, -1), kind: 'code' })
    else {
      const link = token.match(/^\[([^\]]+)]\((https?:\/\/[^\s)]+)\)$/)
      if (link) result.push({ text: link[1], href: link[2], kind: 'link' })
      else result.push({ text: token, kind: 'text' })
    }
    cursor = index + token.length
  }
  if (cursor < text.length) result.push({ text: text.slice(cursor), kind: 'text' })
  return result.length ? result : [{ text, kind: 'text' }]
}

function splitTableRow(line: string): string[] {
  let value = line.trim()
  if (value.startsWith('|')) value = value.slice(1)
  if (value.endsWith('|')) value = value.slice(0, -1)
  const cells: string[] = []
  let current = ''
  let escaped = false
  let inlineCode = false
  for (const character of value) {
    if (escaped) {
      current += character
      escaped = false
      continue
    }
    if (character === '\\') {
      current += character
      escaped = true
      continue
    }
    if (character === '`') inlineCode = !inlineCode
    if (character === '|' && !inlineCode) {
      cells.push(current.trim())
      current = ''
    } else {
      current += character
    }
  }
  cells.push(current.trim())
  return cells
}

function isTableSeparator(line: string, columnCount: number): boolean {
  const cells = splitTableRow(line)
  return cells.length === columnCount && cells.every((cell) => /^:?-{3,}:?$/.test(cell))
}

const blocks = computed<Block[]>(() => {
  const lines = props.content.replace(/\r\n/g, '\n').split('\n')
  const result: Block[] = []
  let index = 0
  while (index < lines.length) {
    const line = lines[index]
    if (!line.trim()) {
      index++
      continue
    }
    const fence = line.match(/^```\s*([\w+#.-]*)/)
    if (fence) {
      const codeLines: string[] = []
      index++
      while (index < lines.length && !lines[index].startsWith('```')) codeLines.push(lines[index++])
      if (index < lines.length) index++
      result.push({ type: 'code', code: codeLines.join('\n'), language: fence[1] })
      continue
    }
    const heading = line.match(/^#{1,4}\s+(.+)/)
    if (heading) {
      result.push({ type: 'heading', parts: parseInline(heading[1]) })
      index++
      continue
    }
    if (/^\s*(?:---+|\*\*\*+)\s*$/.test(line)) {
      result.push({ type: 'rule' })
      index++
      continue
    }
    const tableHeaders = splitTableRow(line)
    if (
      tableHeaders.length >= 2
      && index + 1 < lines.length
      && isTableSeparator(lines[index + 1], tableHeaders.length)
    ) {
      const rows: InlinePart[][][] = []
      index += 2
      while (index < lines.length && lines[index].trim() && lines[index].includes('|')) {
        const cells = splitTableRow(lines[index])
        if (cells.length !== tableHeaders.length) break
        rows.push(cells.map(parseInline))
        index++
      }
      result.push({
        type: 'table',
        headers: tableHeaders.map(parseInline),
        rows,
      })
      continue
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const items: InlinePart[][] = []
      while (index < lines.length && /^\s*[-*]\s+/.test(lines[index])) {
        items.push(parseInline(lines[index].replace(/^\s*[-*]\s+/, '')))
        index++
      }
      result.push({ type: 'list', items })
      continue
    }
    const paragraph: string[] = [line]
    index++
    while (
      index < lines.length
      && lines[index].trim()
      && !/^```/.test(lines[index])
      && !/^#{1,4}\s+/.test(lines[index])
      && !/^\s*[-*]\s+/.test(lines[index])
      && !/^\s*(?:---+|\*\*\*+)\s*$/.test(lines[index])
    ) {
      paragraph.push(lines[index++])
    }
    result.push({ type: 'paragraph', parts: parseInline(paragraph.join('\n')) })
  }
  return result
})
</script>

<template>
  <div class="message-render" :class="tone">
    <template v-for="(block, blockIndex) in blocks" :key="blockIndex">
      <p v-if="block.type === 'paragraph'">
        <template v-for="(part, partIndex) in block.parts" :key="partIndex">
          <strong v-if="part.kind === 'strong'">{{ part.text }}</strong>
          <code v-else-if="part.kind === 'code'">{{ part.text }}</code>
          <a v-else-if="part.kind === 'link'" :href="part.href" target="_blank" rel="noopener noreferrer">{{ part.text }}</a>
          <template v-else>{{ part.text }}</template>
        </template>
      </p>
      <h3 v-else-if="block.type === 'heading'">
        <template v-for="(part, partIndex) in block.parts" :key="partIndex">
          <strong v-if="part.kind === 'strong'">{{ part.text }}</strong>
          <code v-else-if="part.kind === 'code'">{{ part.text }}</code>
          <a v-else-if="part.kind === 'link'" :href="part.href" target="_blank" rel="noopener noreferrer">{{ part.text }}</a>
          <template v-else>{{ part.text }}</template>
        </template>
      </h3>
      <ul v-else-if="block.type === 'list'">
        <li v-for="(item, itemIndex) in block.items" :key="itemIndex">
          <template v-for="(part, partIndex) in item" :key="partIndex">
            <strong v-if="part.kind === 'strong'">{{ part.text }}</strong>
            <code v-else-if="part.kind === 'code'">{{ part.text }}</code>
            <a v-else-if="part.kind === 'link'" :href="part.href" target="_blank" rel="noopener noreferrer">{{ part.text }}</a>
            <template v-else>{{ part.text }}</template>
          </template>
        </li>
      </ul>
      <pre v-else-if="block.type === 'code'"><code :data-language="block.language">{{ block.code }}</code></pre>
      <div v-else-if="block.type === 'table'" class="table-scroll">
        <table>
          <thead>
            <tr>
              <th v-for="(cell, cellIndex) in block.headers" :key="cellIndex">
                <template v-for="(part, partIndex) in cell" :key="partIndex">
                  <strong v-if="part.kind === 'strong'">{{ part.text }}</strong>
                  <code v-else-if="part.kind === 'code'">{{ part.text }}</code>
                  <a v-else-if="part.kind === 'link'" :href="part.href" target="_blank" rel="noopener noreferrer">{{ part.text }}</a>
                  <template v-else>{{ part.text }}</template>
                </template>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, rowIndex) in block.rows" :key="rowIndex">
              <td v-for="(cell, cellIndex) in row" :key="cellIndex">
                <template v-for="(part, partIndex) in cell" :key="partIndex">
                  <strong v-if="part.kind === 'strong'">{{ part.text }}</strong>
                  <code v-else-if="part.kind === 'code'">{{ part.text }}</code>
                  <a v-else-if="part.kind === 'link'" :href="part.href" target="_blank" rel="noopener noreferrer">{{ part.text }}</a>
                  <template v-else>{{ part.text }}</template>
                </template>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <hr v-else />
    </template>
  </div>
</template>

<style scoped>
.message-render { padding: 14px 17px; border: 1px solid #dedbd1; border-radius: 4px 16px 16px 16px; color: #41433e; background: #fbfaf6; font-size: 13px; line-height: 1.8; text-align: left; box-shadow: 0 8px 24px rgba(42,47,41,.04); }
.message-render.live { border-color: #cfd8e9; color: #33415d; background: #f8faff; box-shadow: 0 8px 24px rgba(40,57,91,.05); }
.message-render.user { border: 0; border-radius: 16px 4px 16px 16px; color: #f0f4fb; background: #22385f; }
p, h3, ul, pre, .table-scroll, hr { margin: 0; }
p { white-space: pre-wrap; }
p + p, p + ul, ul + p, h3 + p, h3 + ul, pre + p, p + pre, p + .table-scroll, .table-scroll + p, h3 + .table-scroll, .table-scroll + h3, hr + p, p + hr { margin-top: 9px; }
h3 { color: #2e4d78; font-size: 13px; font-weight: 700; }
.user h3 { color: #fff; }
ul { padding-left: 19px; }
li + li { margin-top: 3px; }
code { padding: 1px 4px; border-radius: 4px; color: #8a5c2c; background: #f0e9dc; font-family: Consolas, 'Cascadia Code', monospace; font-size: .92em; }
.user code { color: #ffe0ad; background: rgba(255,255,255,.1); }
pre { overflow-x: auto; padding: 11px 12px; border-radius: 9px; color: #dce8f8; background: #172944; line-height: 1.55; }
pre code { padding: 0; color: inherit; background: transparent; }
a { color: #2f64a5; text-decoration: underline; text-decoration-color: rgba(47,100,165,.35); text-underline-offset: 3px; overflow-wrap: anywhere; }
a:hover { color: #1f4f8c; text-decoration-color: currentColor; }
.user a { color: #d7e8ff; }
.table-scroll { max-width: 100%; overflow-x: auto; border: 1px solid #d9dee7; border-radius: 10px; background: #fff; }
table { width: 100%; min-width: 560px; border-collapse: collapse; font-size: 12px; line-height: 1.55; }
th, td { padding: 10px 12px; border-right: 1px solid #e3e6ec; border-bottom: 1px solid #e3e6ec; text-align: left; vertical-align: top; }
th:last-child, td:last-child { border-right: 0; }
tbody tr:last-child td { border-bottom: 0; }
th { color: #294a76; background: #eef3f9; font-weight: 700; white-space: nowrap; }
tbody tr:nth-child(even) { background: #f8fafc; }
tbody tr:hover { background: #f2f6fb; }
td:first-child { color: #304e73; font-weight: 650; }
hr { border: 0; border-top: 1px solid #dedbd1; }
</style>
