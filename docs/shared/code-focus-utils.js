const keywordSet = new Set([
  'and',
  'as',
  'assert',
  'break',
  'class',
  'continue',
  'def',
  'del',
  'elif',
  'else',
  'except',
  'False',
  'finally',
  'for',
  'from',
  'global',
  'if',
  'import',
  'in',
  'is',
  'lambda',
  'None',
  'nonlocal',
  'not',
  'or',
  'pass',
  'raise',
  'return',
  'True',
  'try',
  'while',
  'with',
  'yield'
])

const builtinSet = new Set([
  'abs',
  'bool',
  'dict',
  'enumerate',
  'float',
  'int',
  'len',
  'list',
  'max',
  'min',
  'print',
  'range',
  'set',
  'str',
  'sum',
  'tuple',
  'zip'
])

function escapeHtml(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function wrap(className, value) {
  return `<span class="${className}">${escapeHtml(value)}</span>`
}

function highlightPlainCode(value) {
  const pattern = /(@?[A-Za-z_]\w*|\d+(?:\.\d+)?)/g
  let html = ''
  let lastIndex = 0
  let match

  while ((match = pattern.exec(value)) !== null) {
    const token = match[0]
    html += escapeHtml(value.slice(lastIndex, match.index))

    if (token.startsWith('@')) {
      html += wrap('py-decorator', token)
    } else if (keywordSet.has(token)) {
      html += wrap('py-keyword', token)
    } else if (builtinSet.has(token)) {
      html += wrap('py-builtin', token)
    } else if (/^\d/.test(token)) {
      html += wrap('py-number', token)
    } else {
      html += escapeHtml(token)
    }

    lastIndex = match.index + token.length
  }

  html += escapeHtml(value.slice(lastIndex))
  return html
}

export function highlightPython(line) {
  if (!line) return '&nbsp;'

  let html = ''
  let chunk = ''
  let index = 0

  function flushChunk() {
    if (!chunk) return
    html += highlightPlainCode(chunk)
    chunk = ''
  }

  while (index < line.length) {
    const char = line[index]

    if (char === '#') {
      flushChunk()
      html += wrap('py-comment', line.slice(index))
      return html
    }

    if (char === '"' || char === "'") {
      flushChunk()
      const quote = char
      let end = index + 1

      while (end < line.length) {
        if (line[end] === '\\') {
          end += 2
          continue
        }

        if (line[end] === quote) {
          end += 1
          break
        }

        end += 1
      }

      html += wrap('py-string', line.slice(index, end))
      index = end
      continue
    }

    chunk += char
    index += 1
  }

  flushChunk()
  return html
}
