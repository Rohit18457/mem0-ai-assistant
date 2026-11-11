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
    if USE_RICH:
        banner = (
            f"[bold cyan]🧠 AI Memory Assistant[/bold cyan]\n"
            f"[dim]Powered by Mem0 + Groq ({GROQ_MODEL}) + HuggingFace + Qdrant[/dim]\n\n"
            f"[yellow]👤 Active User:[/yellow] [bold white]{user_id}[/bold white]\n"
            f"[dim]Commands: [bold]/memories[/bold] (list all), [bold]/exit[/bold] or [bold]/quit[/bold] (leave)[/dim]"
        )
        console.print(Panel(banner, border_style="cyan", expand=False))
    else:
        print("=" * 60)
        print("🧠 AI Memory Assistant")
        print(f"Powered by Mem0 + Groq ({GROQ_MODEL}) + Qdrant")
        print(f"👤 Active User: {user_id}")
        print("Commands: /memories (list all), /exit (leave)")
        print("=" * 60)


def show_all_memories(assistant: MemoryAssistant):
    memories = assistant.get_all_memories()
    if not memories:
        if USE_RICH:
            console.print("[dim yellow]No memories found for this user yet.[/dim yellow]")
        else:
            print("No memories found for this user yet.")
        return

    if USE_RICH:
        table = Table(title="Saved Memories in Qdrant", border_style="dim")
        table.add_column("#", style="dim", width=4)
        table.add_column("Memory Snippet", style="green")
        table.add_column("Created At", style="dim")

        for idx, item in enumerate(memories, start=1):
            text = item.get("memory", item.get("data", ""))
            created = item.get("created_at", "N/A")
            table.add_row(str(idx), text, str(created)[:19])
        console.print(table)
    else:
        print("\n--- Saved Memories ---")
        for idx, item in enumerate(memories, start=1):
            text = item.get("memory", item.get("data", ""))
            print(f"{idx}. {text}")
        print("-----------------------\n")


def run():
    assistant = MemoryAssistant(user_id=DEFAULT_USER_ID)
    print_banner(DEFAULT_USER_ID)

    while True:
        try:
            if USE_RICH:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
            else:
                user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["/exit", "/quit", "exit", "quit", "q"]:
                if USE_RICH:
                    console.print("[dim cyan]Goodbye! Have a great day![/dim cyan]")
                else:
                    print("Goodbye!")
                break

            if user_input.lower() == "/memories":
                show_all_memories(assistant)
                continue

            # Process chat and get response
            with console.status("[dim cyan]Thinking and querying memory...[/dim cyan]") if USE_RICH else open(sys.devnull):
                ai_text, retrieved = assistant.chat(user_input)

            # Display retrieved memory context if any
            if retrieved:
                if USE_RICH:
                    console.print(f"[dim italic green]💡 Recalled {len(retrieved)} relevant memory snippet(s)[/dim italic green]")
                else:
                    print(f"[💡 Recalled {len(retrieved)} memory snippets]")

            # Display AI response
            if USE_RICH:
                console.print(Panel(Markdown(ai_text), title="[bold green]AI Assistant[/bold green]", border_style="green"))
            else:
                print(f"\nAI: {ai_text}")

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

