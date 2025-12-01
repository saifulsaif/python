

global spam
def scope_testing():

    def local_scope():
        spam = "local spam defined"

    def nonlocal_scope():
        nonlocal spam
        spam = "nonlocal spam defined"
            
    def global_scope():
        spam = "global spam defined"
    spam = "testing the spam"


    local_scope()
    print("After local scope test:", spam)
    nonlocal_scope()
    print("After nonlocal scope test:", spam)
    global_scope()
    print("After global scope test:", spam)

def main():
 scope_testing()

 spam = "testing the spam"
 print("In global scope:", spam)

if __name__ == '__main__':
 main()