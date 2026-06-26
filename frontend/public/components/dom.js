export function domElement(tag, className = "", text = null) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== null && text !== undefined) element.textContent = String(text);
  return element;
}

export function textNodeElement(tag, className = "", text = null) {
  return domElement(tag, className, text ?? "");
}

export function appendPresent(parent, ...items) {
  for (const item of items.flat()) {
    if (item) parent.append(item);
  }
  return parent;
}
