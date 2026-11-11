import sys
from memory_agent import MemoryAssistant
from config import GROQ_MODEL, DEFAULT_USER_ID

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from rich.table import Table
    console = Console()
    USE_RICH = True
except ImportError:
    USE_RICH = False


def print_banner(user_id: str):
    text = (
        f"[bold cyan]🧠 AI Hybrid Memory Assistant (v2.1)[/bold cyan]\n"
        f"[dim]Architecture: Short-Term Buffer + Qdrant Vector DB + Neo4j Graph Store[/dim]\n"
        f"[dim]LLM: Groq ({GROQ_MODEL}) | Embedder: HuggingFace Local (384-d)[/dim]\n\n"
        f"[yellow]👤 Active Profile:[/yellow] [bold white]{user_id}[/bold white]\n"
        f"[dim]Commands: [bold]/help[/bold] | [bold]/memories[/bold] | [bold]/graph[/bold] | [bold]/user <id>[/bold] | [bold]/delete <id>[/bold] | [bold]/exit[/bold][/dim]"
    )
    if USE_RICH:
        console.print(Panel(text, border_style="cyan", expand=False))
    else:
        print("=" * 65)
        print(f"🧠 AI Hybrid Memory Assistant (v2.1) | Active Profile: {user_id}")
        print("Commands: /help, /memories, /graph, /user <id>, /delete <id>, /exit")
        print("=" * 65)


def show_help():
    commands = [
        ("/memories", "List all long-term memories saved in Qdrant Vector DB"),
        ("/graph", "List all memory facts stored in Neo4j Knowledge Graph"),
        ("/delete <id>", "Delete a specific memory by its ID"),
        ("/update <id> <text>", "Update a memory entry with new text"),
        ("/clear", "Delete all memories across Qdrant and Neo4j"),
        ("/user <id>", "Switch to another user profile (or create one)"),
        ("/history", "View short-term working conversation turns"),
        ("/reset", "Clear current session's short-term history"),
        ("/exit or /quit", "Leave the assistant"),
    ]
    if USE_RICH:
        table = Table(title="Available Commands", border_style="cyan")
        table.add_column("Command", style="bold yellow")
        table.add_column("Description", style="white")
        for cmd, desc in commands:
            table.add_row(cmd, desc)
        console.print(table)
    else:
        print("\n--- Available Commands ---")
        for cmd, desc in commands:
            print(f"{cmd:<22} : {desc}")
        print("--------------------------\n")


def show_graph_memories(assistant: MemoryAssistant):
    facts = assistant.get_graph_memories()
    if not facts:
        msg = f"No graph memories found for '{assistant.user_id}' in Neo4j (or Neo4j offline)."
        if USE_RICH:
            console.print(f"[dim yellow]{msg}[/dim yellow]")
        else:
            print(msg)
        return

    if USE_RICH:
        table = Table(title=f"Neo4j Knowledge Graph Facts ({assistant.user_id})", border_style="magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("Graph Node Fact", style="magenta")
        for idx, fact in enumerate(facts, start=1):
            table.add_row(str(idx), fact)
        console.print(table)
    else:
        print(f"\n--- Neo4j Graph Facts ({assistant.user_id}) ---")
        for idx, fact in enumerate(facts, start=1):
            print(f"{idx}. {fact}")
        print("----------------------------------------------\n")


def show_memories(assistant: MemoryAssistant):
    memories = assistant.get_all_memories()
    if not memories:
        msg = "No memories found for this user in Qdrant."
        if USE_RICH:
            console.print(f"[dim yellow]{msg}[/dim yellow]")
        else:
            print(msg)
        return

    if USE_RICH:
        table = Table(title=f"Stored Long-Term Memories ({assistant.user_id})", border_style="dim")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Fact / Memory Content", style="green")
        table.add_column("Created", style="dim", no_wrap=True)

        for item in memories:
            mem_id = item.get("id", "N/A")[:8] + "..."
            text = item.get("memory", item.get("data", ""))
            created = str(item.get("created_at", "N/A"))[:19]
            table.add_row(mem_id, text, created)
        console.print(table)
    else:
        print(f"\n--- Stored Memories ({assistant.user_id}) ---")
        for item in memories:
            print(f"[{item.get('id', 'N/A')[:8]}...] {item.get('memory', item.get('data', ''))}")
        print("-------------------------------------------\n")


def show_history(assistant: MemoryAssistant):
    history = assistant.session_history
    if not history:
        msg = "No session history yet."
        if USE_RICH:
            console.print(f"[dim yellow]{msg}[/dim yellow]")
        else:
            print(msg)
        return

    if USE_RICH:
        table = Table(title="Current Session Working Memory", border_style="dim")
        table.add_column("Turn", style="dim", width=4)
        table.add_column("Role", style="yellow", width=10)
        table.add_column("Message", style="white")
        for idx, turn in enumerate(history, start=1):
            table.add_row(str(idx), turn["role"].capitalize(), turn["content"][:100])
        console.print(table)
    else:
        print("\n--- Session History ---")
        for idx, turn in enumerate(history, start=1):
            print(f"{idx}. [{turn['role'].capitalize()}]: {turn['content']}")
        print("-----------------------\n")


def run():
    assistant = MemoryAssistant(user_id=DEFAULT_USER_ID)
    print_banner(assistant.user_id)

    while True:
        try:
            if USE_RICH:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
            else:
                user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            # Command: Exit
            if user_input.lower() in ["/exit", "/quit", "exit", "quit", "q"]:
                if USE_RICH:
                    console.print("[dim cyan]Goodbye! Have a great day![/dim cyan]")
                else:
                    print("Goodbye!")
                break

            # Command: Help
            if user_input.lower() == "/help":
                show_help()
                continue

            # Command: Show Memories (Qdrant)
            if user_input.lower() == "/memories":
                show_memories(assistant)
                continue

            # Command: Show Graph Memories (Neo4j)
            if user_input.lower() == "/graph":
                show_graph_memories(assistant)
                continue

            # Command: Show Working History
            if user_input.lower() == "/history":
                show_history(assistant)
                continue

            # Command: Reset Session History
            if user_input.lower() == "/reset":
                assistant.clear_session_history()
                msg = "Session short-term history cleared. (Long-term memories retained)"
                if USE_RICH:
                    console.print(f"[dim green]✓ {msg}[/dim green]")
                else:
                    print(f"✓ {msg}")
                continue

            # Command: Switch User
            if user_input.lower().startswith("/user "):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 2:
                    try:
                        assistant.set_user(parts[1])
                        msg = f"Switched active user profile to: '{assistant.user_id}'"
                        if USE_RICH:
                            console.print(f"[bold green]✓ {msg}[/bold green]")
                        else:
                            print(f"✓ {msg}")
                    except ValueError as err:
                        if USE_RICH:
                            console.print(f"[bold red]Error:[/bold red] {err}")
                        else:
                            print(f"Error: {err}")
                continue

            # Command: Delete Memory
            if user_input.lower().startswith("/delete "):
                target_id = user_input.split(maxsplit=1)[1].strip()
                if assistant.delete_memory(target_id):
                    msg = f"Memory '{target_id}' deleted successfully."
                    if USE_RICH:
                        console.print(f"[green]✓ {msg}[/green]")
                    else:
                        print(f"✓ {msg}")
                else:
                    msg = f"Could not delete memory '{target_id}'. Verify ID with /memories."
                    if USE_RICH:
                        console.print(f"[red]✗ {msg}[/red]")
                    else:
                        print(f"✗ {msg}")
                continue

            # Command: Update Memory
            if user_input.lower().startswith("/update "):
                parts = user_input.split(maxsplit=2)
                if len(parts) == 3:
                    target_id, new_text = parts[1].strip(), parts[2].strip()
                    if assistant.update_memory(target_id, new_text):
                        msg = f"Memory '{target_id}' updated."
                        if USE_RICH:
                            console.print(f"[green]✓ {msg}[/green]")
                        else:
                            print(f"✓ {msg}")
                    else:
                        msg = f"Update failed for '{target_id}'."
                        if USE_RICH:
                            console.print(f"[red]✗ {msg}[/red]")
                        else:
                            print(f"✗ {msg}")
                continue

            # Command: Clear All Memories
            if user_input.lower() == "/clear":
                confirm = input(f"Are you sure you want to delete ALL memories for '{assistant.user_id}'? (y/N): ")
                if confirm.lower().strip() == "y":
                    assistant.clear_all_memories()
                    msg = f"All memories for '{assistant.user_id}' have been cleared."
                    if USE_RICH:
                        console.print(f"[yellow]✓ {msg}[/yellow]")
                    else:
                        print(f"✓ {msg}")
                continue

            # Standard Chat Turn
            ai_text, retrieved, saved = assistant.chat(user_input)

            # Context cues
            if retrieved:
                note = f"💡 Recalled {len(retrieved)} relevant memory snippet(s) from past conversations"
                if USE_RICH:
                    console.print(f"[dim italic green]{note}[/dim italic green]")
                else:
                    print(f"[{note}]")

            # Render response
            if USE_RICH:
                console.print(Panel(Markdown(ai_text), title="[bold green]AI Assistant[/bold green]", border_style="green"))
            else:
                print(f"\nAI: {ai_text}")

            # Feedback on newly learned facts
            if saved:
                for f in saved:
                    note = f"📌 Learned new fact: \"{f}\""
                    if USE_RICH:
                        console.print(f"[dim cyan]{note}[/dim cyan]")
                    else:
                        print(note)

        except KeyboardInterrupt:
            print("\nSession ended.")
            break
        except Exception as e:
            if USE_RICH:
                console.print(f"[bold red]Error:[/bold red] {e}")
            else:
                print(f"Error: {e}")


if __name__ == "__main__":
    run()
