export interface Command {
    execute(): void;
    undo(): void;
}

export class CommandManager {
    private static maxHistory = 50;
    private static undoStack: Command[] = [];
    private static redoStack: Command[] = [];

    private static initialized = false;

    /**
     * Initializes the global keyboard shortcuts for Undo/Redo.
     */
    public static initialize() {
        if (this.initialized) return;
        this.initialized = true;
        
        window.addEventListener('keydown', (e) => {
            // Ctrl+Z or Cmd+Z for undo
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
                if (e.shiftKey) {
                    this.redo(); // Ctrl+Shift+Z
                } else {
                    this.undo();
                }
                e.preventDefault();
            }
            // Ctrl+Y or Cmd+Y for redo
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
                this.redo();
                e.preventDefault();
            }
        });
    }

    /**
     * Executes a command and pushes it to the undo stack.
     * Clears the redo stack.
     */
    public static execute(cmd: Command) {
        cmd.execute();
        this.push(cmd);
    }

    /**
     * Pushes a command to the undo stack WITHOUT executing it.
     * Useful when the action (like dragging a slider) has already mutated the state.
     */
    public static push(cmd: Command) {
        this.undoStack.push(cmd);
        if (this.undoStack.length > this.maxHistory) {
            this.undoStack.shift();
        }
        this.redoStack.length = 0;
    }

    public static undo() {
        const cmd = this.undoStack.pop();
        if (cmd) {
            console.log('Undo');
            cmd.undo();
            this.redoStack.push(cmd);
        }
    }

    public static redo() {
        const cmd = this.redoStack.pop();
        if (cmd) {
            console.log('Redo');
            cmd.execute();
            this.undoStack.push(cmd);
        }
    }

    public static clear() {
        this.undoStack.length = 0;
        this.redoStack.length = 0;
    }
}
