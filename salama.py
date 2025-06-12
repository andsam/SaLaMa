
# SaLaMa - Simple Assisted LlAMa Authoring
#
# i.e. a text editor to simultaneously edit and write text while LLM is producing more
# tip; no need to use prompting but just write a beginning, let the LLM continue,
#      edit the output even while it writes, stop it when it starts digressing,
#      and restart the LLM with the edited text - and repeat ad infinitum.
#
# tip; you can use prompts if you want. Many models will play nicely with just **USER** **ASSISTANT** tags,
#      although some will also mix roles and start 'prompting' themselves if you use the tags a lot

# MIT License
# Copyright (c) 2025 Sami Andberg
# See LICENSE file for full license information.

# This might grow into something or not, but it works already as a MVP PoC
# Needs to be run in a directoryt with local llama.cpp installation
#
# Vibe coded with free ChatGPT on 2025-06-06
#
# Still has a known bug in Unicode processing


import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.font as tkfont
import threading
import asyncio
import os
import tempfile


class LLMEditor:

    def update_status(self, status):
        self.root.title(f"SaLaMa Editor - {status}")

    def __init__(self, root):
        self.root = root
        self.update_status("Editor started")

        self.root.bind_all("<Control-Return>", lambda e: self.run_llm_from_editor())
        self.root.bind_all("<Control-BackSpace>", lambda e: self.stop_generation())
        self.root.bind_all("<Control-s>", lambda e: self.save_file())
        self.root.bind_all("<Control-o>", lambda e: self.load_file())

        # Create a frame to contain the text and scrollbar
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create the text widget
        fontti = tkfont.Font(family="fixed", size=22)
        self.text = tk.Text(self.frame, width=150, height=80, font=fontti, wrap=tk.WORD)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a vertical scrollbar
        self.scrollbar = tk.Scrollbar(self.frame, command=self.text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Connect the text widget to the scrollbar
        self.text.config(yscrollcommand=self.scrollbar.set)


        self.button_frame = tk.Frame(root)
        self.button_frame.pack(fill='x')

        self.load_btn = tk.Button(self.button_frame, text="Load", command=self.load_file)
        self.load_btn.pack(side='left')

        self.save_btn = tk.Button(self.button_frame, text="Save", command=self.save_file)
        self.save_btn.pack(side='left')

        self.start_btn = tk.Button(self.button_frame, text="Start LLM", command=self.start_llm)
        self.start_btn.pack(side='left')

        self.model_var = tk.StringVar()
        model_names = [f for f in os.listdir("models") if f.endswith(".gguf") and not f.startswith("ggml-vocab")]
        self.model_var.set(model_names[0] if model_names else "")  # default

        self.model_dropdown = tk.OptionMenu(self.button_frame, self.model_var, *model_names)
        self.model_dropdown.pack(side=tk.LEFT)

        self.stop_btn = tk.Button(self.button_frame, text="Stop LLM", command=self.stop_llm)
        self.stop_btn.pack(side='left')

        self.auto_scroll = tk.BooleanVar(value=True)
        self.scroll_checkbox = tk.Checkbutton(self.button_frame, text="Auto-scroll", variable=self.auto_scroll)
        self.scroll_checkbox.pack(side=tk.LEFT)

        self.exit_btn = tk.Button(self.button_frame, text="Exit", command=root.destroy)
        self.exit_btn.pack(side='right')


        self.llm_task = None
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

    def load_file(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            with open(filepath, 'r') as f:
                self.text.delete(1.0, tk.END)
                self.text.insert(tk.END, f.read())

    def save_file(self):
        filepath = filedialog.asksaveasfilename()
        if filepath:
            with open(filepath, 'w') as f:
                f.write(self.text.get(1.0, tk.END))

    def start_llm(self):
        print("Starting")

        # While preparing input
        self.update_status("Loading "+self.model_var.get()[:-5])

        if self.llm_task and not self.llm_task.done():
            #messagebox.showinfo("Info", "LLM is already running.")
            return

        content = self.text.get("1.0", tk.END).strip()

        self.llm_task = asyncio.run_coroutine_threadsafe(
            self.run_llm(content),
            self.loop
        )

    def stop_llm(self):
        if self.llm_task and not self.llm_task.done():
            self.llm_task.cancel()
            self.llm_task = None
            #messagebox.showinfo("Info", "LLM generation stopped.")
            # On stop or completion
            self.update_status("LLM stopped")

    async def run_llm(self, prompt_text):
        try:

            prompt_len = 0

            fd, prompt_path = tempfile.mkstemp(suffix=".txt", text=True)
            print("Prompt path:", prompt_path)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(prompt_text)
                prompt_len = len(prompt_text)
                f.flush()


            model_path = os.path.join("models", self.model_var.get())
            # all these parameters could be in a UI menu, and some could be model dependent (like temp)
            process = await asyncio.create_subprocess_exec(
                "./llama-cli",                 # path to llama.cpp binary compared to location of this file
                "--ignore-eos",            # keep generating until user stops
                "--main-gpu","0",          # change if needed
                "--n-gpu-layers","6",      # change depending on hardware and GPU memory
                "--no-warmup", 
                "-s","24022405",           # best to keep fixed seed so that new runs don't alter text except what changes have been made in the text itself
                "-t","16",
                "-m", model_path,
                "--file", prompt_path,
                "-c", "60248",             # adjust if needed, should make this selectable in UI or dynamic based on editor text length
                #"--n-predict", "200",     # adjust for how much to generate
                #"--temp", "0.8"           # adjust depending on model
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                close_fds=True
            )

            buffer = ""
            cur_pos = 0
            while True:
                #print("Looping")
                char = await process.stdout.read(1)
                if not char:
                    break
                else:
                    cur_pos+=1
                    if cur_pos > prompt_len+1:
                        self.text.after(0, self.append_text, char)
                        if self.auto_scroll.get():
                            self.text.see("end")
                        # While streaming new text
                        self.update_status("Producing new text")
                    else:
                        # While streaming context
                        #print("Processing context "+str(int(cur_pos/prompt_len+4*100)))
                        self.update_status("Processing context upto "+str(int(cur_pos/(prompt_len+4)*100))+"%")


            await process.wait()

            stderr_output = await process.stderr.read()
            if process.returncode != 0:
                print("LLM Error:", stderr_output.decode())
                #self.text.after(0, lambda: messagebox.showerror("LLM Error", stderr_output.decode()))

        except asyncio.CancelledError:
            print("ASYNCIO cancelled")
            process.kill()
            await process.wait()
            stderr_output = await process.stderr.read()
            if process.returncode != 0:
                print("LLM Error:", stderr_output.decode())
                #self.text.after(0, lambda: messagebox.showerror("LLM Error", stderr_output.decode()))


        except asyncio.CancelledError:
            print("ASYNCIO failed")
            process.kill()
            await process.wait()

        except FileNotFoundError as e:
            print("Error: llama binary not found:", e)
            self.text.after(0, lambda: messagebox.showerror("Error", f"LLM binary not found: {e}"))
            return


    def append_text(self, text):
        # BUG: this has unicode issues with some models
        self.text.insert(tk.END, text)


if __name__ == "__main__":
    root = tk.Tk()
    app = LLMEditor(root)
    root.mainloop()
