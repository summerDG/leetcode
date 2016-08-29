import java.util.LinkedList;
import java.util.Queue;

/**
 * 55. Implement Stack using Queues
 */
public class MyStack {
    Queue<Integer> q1 = new LinkedList<Integer>();
    Queue<Integer> q2 = new LinkedList<Integer>();
    int front;
    // Push element x onto stack.
    public void push(int x) {
        front = x;
        q1.add(x);
    }

    // Removes the element on top of the stack.
    public void pop() {
        boolean f = true;
        if(empty()) return;
        if (!q1.isEmpty()){
            while(f){
                int i = q1.poll();
                if(q1.isEmpty()) f = false;
                else{
                    q2.add(i);
                    front = i;
                }
            }
        }else {
            while(f){
                int i = q2.poll();
                if(q2.isEmpty()) f = false;
                else{
                    q1.add(i);
                    front = i;
                }
            }
        }
    }

    // Get the top element.
    public int top() {
        return front;
    }

    // Return whether the stack is empty.
    public boolean empty() {
        return q1.isEmpty() && q2.isEmpty();
    }
}
