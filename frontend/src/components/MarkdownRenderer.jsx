import React, { useEffect, useId } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import mermaid from "mermaid";

// Initialize Mermaid once
mermaid.initialize({ startOnLoad: false, theme: "default" });

function MermaidBlock({ code }) {
  const id = useId().replaceAll(":", "");
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const { svg } = await mermaid.render(`m-${id}`, code);
        const el = document.getElementById(`m-${id}`);
        if (el && mounted) {
          el.innerHTML = svg;
        }
      } catch (e) {
        const el = document.getElementById(`m-${id}`);
        if (el && mounted) {
          el.textContent = "Mermaid rendering error.";
        }
      }
    })();
    return () => {
      mounted = false;
    };
  }, [code, id]);
  return <div id={`m-${id}`} className="w-full overflow-x-auto my-2" />;
}

export default function MarkdownRenderer({ children }) {
  return (
    <div className="prose max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={{
          code({ node, inline, className, children, ...props }) {
            const lang = (className || "").replace("language-", "");
            const content = String(children || "");
            if (!inline && lang === "mermaid") {
              return <MermaidBlock code={content} />;
            }
            return (
              <code className="bg-gray-100 px-1.5 py-0.5 rounded" {...props}>
                {children}
              </code>
            );
          },
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
