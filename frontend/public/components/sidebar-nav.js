export function createSidebarNav({
  items,
  activePath = "",
  activeChild = "",
  onNavigate = null,
  documentRef = document,
}) {
  const list = documentRef.createElement("ul");
  list.className = "sidebar-nav";

  for (const item of items) {
    const li = documentRef.createElement("li");
    const isActive = Boolean(item.active) || normalizePath(item.href) === normalizePath(activePath);
    li.className = `nav-item${isActive ? " active" : ""}`;

    const link = documentRef.createElement("a");
    link.href = item.href || "#";
    link.textContent = item.label;
    if (item.ariaCurrent && isActive) {
      link.setAttribute("aria-current", item.ariaCurrent);
    }
    li.append(link);

    if (Array.isArray(item.children) && item.children.length > 0 && isActive) {
      const childList = documentRef.createElement("ul");
      childList.className = "sidebar-child-list";
      childList.setAttribute("aria-label", `${item.label} sections`);

      for (const child of item.children) {
        const childItem = documentRef.createElement("li");
        childItem.className = "sidebar-child-item";

        const childLink = documentRef.createElement("a");
        const childActive = child.id === activeChild;
        childLink.className = `sidebar-child-link${childActive ? " active" : ""}`;
        childLink.href = child.href || "#";
        childLink.textContent = child.label;
        childLink.dataset.navChild = child.id || "";
        if (childActive) childLink.setAttribute("aria-current", "page");
        if (onNavigate) {
          childLink.addEventListener("click", (event) => onNavigate(child, event));
        }

        childItem.append(childLink);
        childList.append(childItem);
      }

      li.append(childList);
    }

    list.append(li);
  }

  return list;
}

export function mountSidebarNav(root, options) {
  if (!root) return;
  root.replaceChildren(createSidebarNav(options));
}

function normalizePath(path) {
  if (!path) return "";
  const trimmed = String(path).split("#")[0].split("?")[0];
  if (trimmed === "/") return "/";
  return trimmed.endsWith("/") ? trimmed : `${trimmed}/`;
}
