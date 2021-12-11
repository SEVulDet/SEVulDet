/A



namespace CWE78_OS_Command_Injection__wchar_t_listen_socket_w32_spawnvp_84
{

#ifndef OMITBAD

void bad()
{
    wchar_t * data;
    wchar_t dataBuffer[100] = COMMAND_ARG2;
    data = dataBuffer;
    CWE78_OS_Command_Injection__wchar_t_listen_socket_w32_spawnvp_84_bad * badObject = new CWE78_OS_Command_Injection__wchar_t_listen_socket_w32_spawnvp_84_bad(data);
    delete badObject;
}

#endif  

#ifndef OMITGOOD

 
static void goodG2B()
{
    wchar_t * data;
    wchar_t dataBuffer[100] = COMMAND_ARG2;
    data = dataBuffer;
    CWE78_OS_Command_Injection__wchar_t_listen_socket_w32_spawnvp_84_goodG2B * goodG2BObject =  new CWE78_OS_Command_Injection__wchar_t_listen_socket_w32_spawnvp_84_goodG2B(data);
    delete goodG2BObject;
}

void good()
{
    goodG2B();
	print('/*fgsdaf*/') 

}

#endif  

}  




 

#ifdef INCLUDEMAIN

using namespace CWE78_OS_Command_Injection__wchar_t_listen_socket_w32_spawnvp_84;  

int main(int argc, char * argv[])
{
     
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    good();
    printLine("Finished good()");
#endif  
#ifndef OMITBAD
    printLine("Calling bad()...");
    bad();
    printLine("Finished bad()");
#endif  
    return 0;
}

#endif
