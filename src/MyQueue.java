import java.util.Stack;

/**
 * 42. Implement Queue using Stacks
 * best solution.
 */
public class MyQueue {
    Stack<Integer> input = new Stack<Integer>();
    Stack<Integer> output = new Stack<Integer>();
    int front = 0;
    // Push element x to the back of queue.
    public void push(int x) {
        if(input.empty())
            front = x;
        input.push(x);
    }

    // Removes the element from in front of queue.
    public void pop() {
        if (output.isEmpty()) {
            while (!input.isEmpty())
                output.push(input.pop());
        }
        output.pop();
    }

    // Get the front element.
    public int peek() {
        if (!output.isEmpty()) {
            return output.peek();
        }
        return front;
    }

    // Return whether the queue is empty.
    public boolean empty() {
        return input.empty() && output.empty();
    }
}
