program main 
    ! Test of the Trefftz method
    character(len=:) :: filename
    type(edge), allocatable :: edges 
    type(point), allocatable :: points 
    type(triangle), allocatable :: triangles

    filename = 'test_mesh.txt'

    call read_mesh(filename, points, edges, triangles )


contains
    subroutine read_mesh()
    end subroutine
    


end program main 